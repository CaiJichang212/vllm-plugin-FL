"""Exact host-dataflow tests against frozen git reference, without GPU imports.

Extract real caller and native consumer ASTs; mock only projection/device calls.
These tests do not establish GPU compatibility or measured model benefit.
"""
import ast
import copy
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS
from unittest.mock import patch
import unittest

import torch
from torch.utils._python_dispatch import TorchDispatchMode

ROOT = Path(__file__).resolve().parents[1]
BASE = '5f39700683751f382e099bd638df4e3dddbc84b3'
FILE = 'vllm_fl/models/minicpm5_rope_kv.py'
ATTN = 'vllm_fl/dispatch/backends/vendor/metax/impl/attention/flash_attn.py'
BASE_SOURCE = subprocess.check_output(['git', '-C', str(ROOT), 'show', BASE+':'+FILE], text=True)
SOURCE = (ROOT / FILE).read_text()
TYPES = NS(DECODER='decoder', ENCODER='encoder', ENCODER_ONLY='encoder-only')


def node(source, name, cls=None):
    body = ast.parse(source).body
    if cls:
        body = next(n for n in body if isinstance(n, ast.ClassDef) and n.name == cls).body
    return copy.deepcopy(next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name))


def load(n, env):
    exec(compile(ast.fix_missing_locations(ast.Module(body=[n], type_ignores=[])), FILE, 'exec'), env)
    return env[n.name]


class Views(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(str(func))
        return func(*args, **(kwargs or {}))


class SpecialTensor(torch.Tensor):
    pass


class Harness:
    def __init__(self, source, direct=True):
        self.events, self.queries, self.kv_args = [], [], []
        self.cache = torch.full((2, 3, 16, 2, 128), -17., device='cpu')
        def decode(fn, q, k, v, out, *args):
            self.events.append('decode')
            if direct:
                out.copy_(q + k.sum() + v.sum())
            return direct
        def fallback(**kw):
            self.events.append('decode_fallback')
            return kw['q'] + kw['k_cache'].sum() + kw['v_cache'].sum()
        def forbidden(*a, **kw):
            raise AssertionError('unexpected K/V consumer or insertion')
        env = dict(torch=torch, FlashAttentionMetadata=object, AttentionType=TYPES,
                   try_decode_into=decode, flash_attn_with_kvcache=fallback,
                   reshape_query_for_spec_decode=lambda q,n:q,
                   reshape_attn_output_for_spec_decode=lambda q:q,
                   reshape_and_cache_flash=forbidden)
        native = load(node((ROOT/ATTN).read_text(), 'forward', 'FlashAttentionImpl'), env)
        # The caller must see an exact native type and bound function identity.
        self.Impl = type('FlashAttentionImpl', (), {'forward': native})
        self.impl = self.Impl()
        self.impl.__dict__.update(attn_type='decoder', kv_cache_dtype='auto',
            dcp_world_size=1, kv_sharing_target_layer_name=None, num_kv_heads=2,
            scale=.5, sliding_window=(-1,-1), alibi_slopes=None, logits_soft_cap=0., sinks=None)
        self.md = NS(num_prefills=0, num_prefill_tokens=0, use_cascade=False,
            num_decodes=4, num_decode_tokens=4, num_actual_tokens=4,
            query_start_loc=torch.tensor([0,1,2,3,4]), seq_lens=None,
            max_query_len=1, max_seq_len=16, block_table=None, scheduler_metadata=None,
            decode_block_table=None, decode_seq_lens=None, slot_mapping=None)
        self.layer = NS(impl=self.impl)
        self.qkv = None
        self.fail = None
        def projection(*args):
            self.events.append('projection')
            if self.fail == 'unsupported': return None
            if self.fail == 'projection': raise ValueError('submitted failure')
            # Emulate complete native cache writes, including untouched sentinels.
            m = self.qkv.shape[0]
            self.cache[0,0,:m].copy_(self.qkv[:,2048:2304].view(m,2,128))
            self.cache[1,0,:m].copy_(self.qkv[:,2304:].view(m,2,128))
            return self.qkv
        self.module = NS(try_qkv_rope_cache=projection)
        def transfer(fn):
            def call(*a, **kw):
                self.events.append('wait')
                result = fn(*a, **kw)
                self.events.append('save')
                return result
            return call
        env = dict(torch=torch, FlashAttentionImpl=self.Impl, _QKV_NATIVE_FORWARD=native,
                   AttentionType=TYPES, maybe_transfer_kv_layer=transfer,
                   get_attention_context=lambda name:(self.md,self.layer,self.cache,None),
                   _qkv_observe=False, trace=lambda *args:None)
        self.fn = load(node(source, 'fused_qkv_attention'), env)
        self.model = NS(qkv_proj=NS(weight=None), rotary_emb=NS(cos_sin_cache=None))

    def run(self, qkv):
        self.qkv = qkv
        m = qkv.shape[0]
        self.md.num_decodes = self.md.num_decode_tokens = self.md.num_actual_tokens = m
        store = torch.full((m+2,16,256), -99., device='cpu')
        out = store[1:m+1,:,::2]
        with patch.dict(sys.modules, {'flag_gems.runtime.backend._metax.fused.qkv_rope_cache':self.module}):
            with Views() as counts:
                self.fn(self.model, None, None, out, 'layer')
        return store, counts.ops


class UnusedKVTests(unittest.TestCase):
    def qkv(self, m, strided=False, bias=0):
        raw = torch.arange((m+2)*5120, dtype=torch.float32, device='cpu').reshape(m+2,5120)
        return raw[1:m+1,::2] + bias if not strided else raw[1:m+1,::2]

    def test_native_consumer_exact_outputs_cache_and_view_reduction(self):
        for m in (1,2,3,4):
            for strided in (False,True):
                for direct in (False,True):
                    with self.subTest(m=m,strided=strided,direct=direct):
                        a,b = Harness(BASE_SOURCE,direct),Harness(SOURCE,direct)
                        qkv = self.qkv(m,strided)
                        x,ops_a = a.run(qkv); y,ops_b = b.run(qkv)
                        torch.testing.assert_close(x,y,rtol=0,atol=0)
                        torch.testing.assert_close(a.cache,b.cache,rtol=0,atol=0)
                        self.assertEqual(a.events,b.events)
                        self.assertTrue(torch.all(y[0] == -99))
                        self.assertTrue(torch.all(y[-1] == -99))
                        self.assertTrue(torch.all(y[:,:,1::2] == -99))
                        self.assertEqual(ops_a.count('aten.split_with_sizes.default'),1)
                        self.assertEqual(ops_b.count('aten.split_with_sizes.default'),0)
                        self.assertEqual(ops_a.count('aten.view.default')-ops_b.count('aten.view.default'),2)

    def test_profile_hit_observes_decorated_original_without_replacement(self):
        h=Harness(SOURCE)
        hits=[]
        def profile(frame,event,arg):
            if event == 'call' and frame.f_code.co_qualname == 'fused_qkv_attention':
                hits.append(frame.f_code.co_filename)
        self.assertIsNone(sys.getprofile())
        try:
            sys.setprofile(profile)
            _,ops=h.run(self.qkv(1))
        finally:
            sys.setprofile(None)
        self.assertEqual(hits,[FILE])
        self.assertNotIn('aten.split_with_sizes.default',ops)

    def test_bfloat16_exact_outputs_and_cache(self):
        a,b=Harness(BASE_SOURCE),Harness(SOURCE)
        a.cache=a.cache.to(torch.bfloat16); b.cache=b.cache.to(torch.bfloat16)
        qkv=self.qkv(4,True).to(torch.bfloat16)
        x,_=a.run(qkv); y,_=b.run(qkv)
        torch.testing.assert_close(x,y,rtol=0,atol=0)
        torch.testing.assert_close(a.cache,b.cache,rtol=0,atol=0)

    def test_q_mapping_offsets_and_strides(self):
        for m in (1,2,3,4):
            qkv = self.qkv(m,True)
            a = qkv.split([2048,256,256],dim=-1)[0].view(-1,16,128)
            b = qkv.narrow(-1,0,2048).view(-1,16,128)
            torch.testing.assert_close(a,b,rtol=0,atol=0)
            self.assertEqual(a.stride(),b.stride())
            self.assertEqual(a.storage_offset(),b.storage_offset())
            self.assertEqual(a.data_ptr(),b.data_ptr())

    def test_custom_consumer_and_dynamic_invalidation(self):
        h = Harness(SOURCE)
        self.assertNotIn('aten.split_with_sizes.default',h.run(self.qkv(1))[1])
        seen=[]
        def custom(layer,q,k,v,*args,**kw):
            seen.append((k.clone(),v.clone()))
            self.assertIsNotNone(k); self.assertIsNotNone(v)
        h.impl.forward = custom
        self.assertIn('aten.split_with_sizes.default',h.run(self.qkv(2))[1])
        self.assertEqual(len(seen),1)
        del h.impl.forward
        self.assertNotIn('aten.split_with_sizes.default',h.run(self.qkv(3))[1])
        h.Impl.forward = custom
        self.assertIn('aten.split_with_sizes.default',h.run(self.qkv(4))[1])

    def test_unsupported_modes_and_subclasses_keep_complete_views(self):
        for field,value in [('attn_type','encoder'),('dcp_world_size',2),
                            ('kv_cache_dtype','fp8'),('use_cascade',True),
                            ('num_prefills',1),('num_prefill_tokens',1),('subclass',True)]:
            h = Harness(SOURCE)
            seen=[]
            # Retain function identity but inspect arguments instead of executing unsupported native paths.
            def record(self,layer,q,k,v,*args,**kw):
                seen.append((k,v))
            h.Impl.forward = record
            # Bind new verified identity for this harness only; mode/type checks still independently apply.
            # Access closure environment of the extracted wrapped caller.
            fn = next(c.cell_contents for c in h.fn.__closure__ if callable(c.cell_contents))
            fn.__globals__['_QKV_NATIVE_FORWARD'] = record
            if hasattr(h.impl,field): setattr(h.impl,field,value)
            elif hasattr(h.md,field): setattr(h.md,field,value)
            if field == 'subclass': h.impl.__class__ = type('Subclass',(h.Impl,),{})
            _,ops = h.run(self.qkv(2))
            self.assertIn('aten.split_with_sizes.default',ops)
            self.assertIsNotNone(seen[0][0]); self.assertIsNotNone(seen[0][1])
        h=Harness(SOURCE)
        self.assertIn('aten.split_with_sizes.default',h.run(self.qkv(1).as_subclass(SpecialTensor))[1])

    def test_consecutive_replacements_use_fresh_data(self):
        h=Harness(SOURCE)
        for m,bias in [(4,0),(1,12),(3,-5),(2,33)]:
            qkv=self.qkv(m,bias=bias)
            out,_=h.run(qkv)
            k,v=h.cache.unbind(0)
            expected=qkv[:,:2048].view(m,16,128)+k.sum()+v.sum()
            torch.testing.assert_close(out[1:m+1,:,::2],expected,rtol=0,atol=0)

    def test_no_retry_or_save_after_submitted_failure(self):
        for fail in ('unsupported','projection'):
            h=Harness(SOURCE); h.fail=fail
            with self.assertRaises((ValueError,RuntimeError)):
                h.run(self.qkv(1))
            self.assertEqual(h.events,['wait','projection'])
        h=Harness(SOURCE)
        def broken(*a,**kw): raise ValueError('consumer failure')
        h.impl.forward=broken
        with self.assertRaises(ValueError): h.run(self.qkv(1))
        self.assertEqual(h.events,['wait','projection'])

    def test_only_view_preparation_changed(self):
        a,b=ast.parse(BASE_SOURCE),ast.parse(SOURCE)
        b.body=[n for n in b.body if not (isinstance(n,ast.Assign) and any(
            isinstance(t,ast.Name) and t.id=='_QKV_NATIVE_FORWARD' for t in n.targets))]
        fn=next(n for n in b.body if isinstance(n,ast.FunctionDef) and n.name=='fused_qkv_attention')
        for i,n in enumerate(fn.body):
            if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='impl' for t in n.targets):
                self.assertIsInstance(fn.body[i+1],ast.If)
                fn.body[i:i+2]=fn.body[i+1].orelse
                break
        self.assertEqual(ast.dump(a),ast.dump(b))

if __name__ == '__main__':
    unittest.main()
