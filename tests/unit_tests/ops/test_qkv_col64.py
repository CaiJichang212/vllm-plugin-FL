"""Column-tile candidate. Test structure adapted from 22011906e4afdfa810b38646900cb767824d65f4 (Apache-2.0); mechanism and evidence are independent."""
import ast
import contextlib
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[3]
SOURCE = REPO / 'vllm_fl/models/minicpm5_rope_kv.py'
GEMS = next((REPO.parent / name for name in ('FlagGems', 'gems')
             if (REPO.parent / name / 'src').is_dir()), REPO.parent / 'gems')
KERNEL = GEMS / 'src/flag_gems/runtime/backend/_metax/fused/qkv_rope_cache.py'
MODULE = 'flag_gems.runtime.backend._metax.fused.qkv_rope_cache'


def extracted(path, names, ns):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if len(nodes) != len(names): raise AssertionError('missing committed function')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
    return ns


class Tensor:
    shape = (4, 2048)


class HostContract(unittest.TestCase):
    def test_exact_launch_contract_and_unsupported_never_submit(self):
        events = []
        state = NS(support=True, suppressed=False)
        class Kernel:
            def __getitem__(self, grid):
                def run(*args, **kwargs):
                    events.append((grid, args, kwargs))
                    return None if state.suppressed else object()
                return run
        output = object()
        ns = dict(supported=lambda *a: state.support,
                  torch=NS(empty=lambda *a, **kw: output),
                  torch_device_fn=NS(device=lambda _: contextlib.nullcontext()),
                  qkv_rope_cache_kernel=Kernel())
        extracted(KERNEL, {'try_qkv_rope_cache_m4_col64'}, ns)
        x=NS(shape=(4,2048),dtype='bf16',device='cuda',stride=lambda i:2048)
        w=NS(stride=lambda i:2048)
        pos=NS(stride=lambda i:1)
        table=NS(shape=(128,128),stride=lambda:(128,1))
        cache=NS(shape=(2,16,16,2,128))
        slots=NS(stride=lambda i:1,numel=lambda:4)
        invoke=lambda:ns['try_qkv_rope_cache_m4_col64'](x,w,pos,table,cache,slots)
        self.assertIs(invoke(),output)
        self.assertEqual(events[0],((40,),(x,w,output,pos,table,cache,slots,
                                         4,2048,2048,1,128,1,1,4,16,128),
                                   dict(BLOCK_N=64,num_warps=4,num_stages=4,enable_fp_fusion=True)))
        for rows in (1,2,3,5,32):
            x.shape=(rows,2048);self.assertIsNone(invoke())
        x.shape=(4,2048);state.support=False;self.assertIsNone(invoke())
        self.assertEqual(len(events),1)
        state.support=True;state.suppressed=True
        with self.assertRaisesRegex(RuntimeError,'state is unknown'):invoke()
        self.assertEqual(len(events),2)

    def test_fallback_boundaries_and_no_retry_after_submission(self):
        events=[];state=NS(support=True,error=None)
        op=ModuleType(MODULE)
        def new(*args):
            events.append('col64')
            if state.error: raise state.error
            return 'new' if state.support else None
        op.try_qkv_rope_cache_m4_col64=new
        op.try_qkv_rope_cache=lambda *a:events.append('original') or 'old'
        ns=extracted(SOURCE,{'project_qkv_decode_col64','original_qkv_projection'},dict(torch=NS(Tensor=Tensor)))
        def invoke(x=Tensor(),phase='decode',name='model.layers.0.self_attn.attn'):
            return ns['project_qkv_decode_col64'](x,None,None,None,None,None,phase,name)
        with patch.dict(sys.modules,{MODULE:op}):
            self.assertEqual(invoke(),'new')
            for name in (None,'bad','model.layers.42.self_attn.attn','model.layers.-1.self_attn.attn'):
                before=len(events);self.assertEqual(invoke(name=name),'old')
                self.assertEqual(events[before:],['original'])
            for phase in ('prefill','mixed','profile'):
                before=len(events);self.assertEqual(invoke(phase=phase),'old')
                self.assertEqual(events[before:],['original'])
            for rows in (1,3,5):
                x=Tensor();x.shape=(rows,2048)
                self.assertEqual(invoke(x=x),'old')
            self.assertEqual(invoke(x=type('Subclass',(Tensor,),{})()),'old')
            state.support=False;self.assertEqual(invoke(),'old')
            self.assertEqual(events[-2:],['col64','original'])
            state.error=RuntimeError('submitted kernel error');before=len(events)
            with self.assertRaisesRegex(RuntimeError,'submitted'):invoke()
            self.assertEqual(events[before:],['col64'])

    def test_real_outer_and_actual_branch_have_all_layer_combinations(self):
        op=ModuleType(MODULE);op.try_qkv_rope_cache_m4_col64=lambda *a:'qkv'
        op.try_qkv_rope_cache=lambda *a:None
        ns=extracted(SOURCE,{'project_qkv_decode_col64','original_qkv_projection'},dict(torch=NS(Tensor=Tensor)))
        tree=ast.parse(SOURCE.read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RopeKVAttention')
        cls.bases=[];cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward']
        branch=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='project_qkv_decode_col64')
        line=next(n.lineno for n in branch.body if isinstance(n,ast.Return) and isinstance(n.value,ast.Name) and n.value.id=='qkv')
        output=NS(view=lambda *a:'flat')
        helper=ns['project_qkv_decode_col64']
        def fused(self,positions,hidden,out,name):
            self_test.assertEqual(helper(hidden,None,None,None,None,None,'decode',name),'qkv')
            return True
        self_test=self
        # The helper retains its original tensor facade after the outer stub.
        outer=dict(torch=NS(empty=lambda *a,**kw:output),qkv_fallback_reason=lambda *a:None,
                   get_attention_context=lambda name:(None,None,None,None),fused_qkv_attention=fused)
        exec(compile(ast.Module(body=[cls],type_ignores=[]),str(SOURCE),'exec'),outer)
        calls=[];seen=set()
        def profiler(frame,event,arg):
            if event=='call' and frame.f_code.co_qualname=='RopeKVAttention.forward':calls.append(1)
        def tracer(frame,event,arg):
            if event=='line' and frame.f_code.co_qualname=='project_qkv_decode_col64' and frame.f_lineno==line:
                d=frame.f_locals;seen.add((tuple(d['x'].shape),d['phase'],d['layer']))
            return tracer
        oldp,oldt=sys.getprofile(),sys.gettrace()
        try:
            with patch.dict(sys.modules,{MODULE:op}):
                sys.setprofile(profiler);sys.settrace(tracer)
                for layer in range(42):
                    obj=NS(attn=NS(layer_name=f'model.layers.{layer}.self_attn.attn'),q_size=2048,
                           o_proj=lambda x:('result',None))
                    x=Tensor();x.device='cuda';x.dtype='bf16'
                    self.assertEqual(outer['RopeKVAttention'].forward(obj,None,x),'result')
        finally:
            sys.setprofile(oldp);sys.settrace(oldt)
        self.assertEqual(len(calls),42)
        self.assertEqual(seen,{((4,2048),'decode',i) for i in range(42)})

    def test_kernel_arithmetic_unchanged_and_transfer_boundary_preserved(self):
        tree=ast.parse(KERNEL.read_text())
        kernel=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='qkv_rope_cache_kernel')
        # Frozen base AST identity: this candidate only changes launch column extent.
        self.assertEqual(hashlib.sha256(ast.dump(kernel,include_attributes=False).encode()).hexdigest(),
                         '8fbc11c9b22eecd14a39b98ef3685f59a72b63509698f140092eac1a4f603290')
        tree=ast.parse(SOURCE.read_text())
        fused=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='fused_qkv_attention')
        self.assertEqual([n.id for n in fused.decorator_list],['maybe_transfer_kv_layer'])
        self.assertFalse(any(isinstance(n,ast.Try) for n in ast.walk(fused)))
        calls=[n for n in ast.walk(fused) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='project_qkv_decode_col64']
        self.assertEqual(len(calls),1)
        self.assertEqual(ast.unparse(calls[0].args[-2]),"'decode' if metadata.num_prefills == 0 else 'mixed'")
        # Independent column ownership: every projection/rope pair is written once.
        def columns(block):
            cols=[]
            for cta in range(2560//block):
                for lane in range(block):
                    pair=cta*(block//2)+lane//2
                    cols.append((pair//64)*128+pair%64+(lane%2)*64)
            return cols
        for block in (32,64):
            cols=columns(block)
            self.assertEqual(len(cols),len(set(cols)))
            self.assertEqual(set(cols),set(range(2560)))


class DeviceContract(unittest.TestCase):
    def test_all_42_real_weight_qkv_and_cache_exact(self):
        import torch
        from safetensors import safe_open
        protocol=json.loads(Path(os.environ['C500_FROZEN_PROTOCOL']).read_text())
        model=Path(protocol['model'])
        index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
        op=importlib.import_module(MODULE)
        torch.manual_seed(20261021)
        with torch.inference_mode():
            for layer in range(42):
                pieces=[]
                for part in ('q','k','v'):
                    name=f'model.layers.{layer}.self_attn.{part}_proj.weight'
                    with safe_open(model/index[name],framework='pt',device='cpu') as handle:
                        pieces.append(handle.get_tensor(name))
                weight=torch.cat(pieces).to(device='cuda',dtype=torch.bfloat16)
                del pieces
                for count,offset in ((4,0),(2,8),(0,8)):
                    raw=torch.randn((offset+4*2048,),device='cuda',dtype=torch.bfloat16)
                    x=raw[offset:].view(4,2048)
                    positions=torch.tensor([0,1,31,127],device='cuda',dtype=torch.int64)
                    table=torch.randn((128,128),device='cuda',dtype=torch.float32)*.1
                    slots=torch.tensor([-1,0,31,255][:count],device='cuda',dtype=torch.int64)
                    original=torch.randn((2,16,16,2,128),device='cuda',dtype=torch.bfloat16)
                    a,b=original.clone(),original.clone()
                    saved=[t.clone() for t in (x,weight,positions,table,slots)]
                    expected=op.try_qkv_rope_cache(x,weight,positions,table,a,slots)
                    torch.cuda.synchronize()
                    actual=op.try_qkv_rope_cache_m4_col64(x,weight,positions,table,b,slots)
                    torch.cuda.synchronize()
                    self.assertIsNotNone(actual)
                    self.assertTrue(torch.isfinite(actual).all().item(),(layer,count))
                    self.assertTrue(torch.equal(actual,expected),('qkv',layer,count,offset))
                    self.assertTrue(torch.equal(a,b),('cache',layer,count,offset))
                    # Independent inactive slots must retain initial values.
                    active={s for s in [-1,0,31,255][:count] if s>=0}
                    for slot in range(256):
                        if slot not in active:
                            self.assertTrue(torch.equal(b[:,slot//16,slot%16],original[:,slot//16,slot%16]))
                    for t,old in zip((x,weight,positions,table,slots),saved):
                        self.assertTrue(torch.equal(t,old))
                # Valid baseline M3 must fall back before a col64 submission.
                self.assertIsNone(op.try_qkv_rope_cache_m4_col64(x[:3],weight,
                                  positions[:3],table,b,slots))
                del weight


if __name__=='__main__':unittest.main()
