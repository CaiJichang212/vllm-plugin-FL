"""Exact call-local Q/K/V view contract; GPU case runs only in owned worker."""
import ast
from pathlib import Path
from types import SimpleNamespace as N
import unittest

SOURCE = Path(__file__).resolve().parents[3] / 'vllm_fl/models/minicpm5_rope_kv.py'


def functions(torch):
    tree = ast.parse(SOURCE.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in
             ('_direct_qkv_views', '_original_qkv_views')]
    ns = dict(torch=torch)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), ns)
    return ns


class HostContract(unittest.TestCase):
    def test_exact_metadata_offsets_and_reject_domain(self):
        class Tensor:
            ndim = 2
            shape = (4, 2560)
            row = 2560
            inner = 1
            offset = 37
            def stride(self, dim): return (self.row, self.inner)[dim]
            def storage_offset(self): return self.offset
            def as_strided(self, shape, stride, offset): return (shape, stride, offset)
        fn = functions(N(Tensor=Tensor))['_direct_qkv_views']
        x = Tensor()
        for row in (2560, 2601):
            for offset in (0, 37):
                x.row, x.offset = row, offset
                q, k, v = fn(x)
                self.assertEqual(q, ((4, 16, 128), (row, 128, 1), offset))
                self.assertEqual(k, ((4, 2, 128), (row, 128, 1), offset + 2048))
                self.assertEqual(v, ((4, 2, 128), (row, 128, 1), offset + 2304))
                # Independently enumerate every address, including padding/offset.
                for view, start, heads in ((q, 0, 16), (k, 2048, 2), (v, 2304, 2)):
                    actual = [view[2]+m*view[1][0]+h*128+d
                              for m in range(4) for h in range(heads) for d in range(128)]
                    expected = [offset+m*row+start+c for m in range(4) for c in range(heads*128)]
                    self.assertEqual(actual, expected)
        for shape, row, inner in (((1,2560),2560,1), ((2,2560),2560,1),
                                  ((3,2560),2560,1), ((5,2560),2560,1),
                                  ((4,2559),2560,1), ((4,2560),2559,1),
                                  ((4,2560),5120,2), ((4,2560),-2560,1)):
            x.shape, x.row, x.inner = shape, row, inner
            self.assertIsNone(fn(x))
        class Subclass(Tensor): pass
        self.assertIsNone(fn(Subclass()))

    def test_consumer_values_and_submission_order(self):
        import sys
        import types
        from unittest.mock import patch
        tree = ast.parse(SOURCE.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'fused_qkv_attention')
        events = []
        q, k, v = object(), object(), object()
        state = N(views=(q,k,v), output=object(), error=None)
        op = types.ModuleType('flag_gems.runtime.backend._metax.fused.qkv_rope_cache')
        def launch(*args):
            events.append('launch')
            if state.error: raise state.error
            return state.output
        op.try_qkv_rope_cache = launch
        def transfer(f):
            def wrapped(*args):
                events.append('wait')
                result=f(*args)
                events.append('save')
                return result
            return wrapped
        def original(value): events.append('original_views'); return q,k,v
        def attention(*args, **kwargs):
            self.assertEqual(args[1:4], (q,k,v))
            self.assertTrue(kwargs['cache_already_written'])
            events.append('attention')
        layer=N(impl=N(forward=attention))
        cache=N(unbind=lambda _: ('Kcache','Vcache'))
        ns=dict(maybe_transfer_kv_layer=transfer,
                get_attention_context=lambda _: (N(slot_mapping=None), layer, cache, None),
                _direct_qkv_views=lambda _: state.views,
                _original_qkv_views=original,_qkv_observe=False,trace=lambda *a: None)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(SOURCE),'exec'),ns)
        with patch.dict(sys.modules,{op.__name__:op}):
            invoke=lambda name='model.layers.41.self_attn.attn': ns['fused_qkv_attention'](
                N(qkv_proj=N(weight=None),rotary_emb=N(cos_sin_cache=None)),None,None,None,name)
            for name, views, expected in [('model.layers.41.self_attn.attn',(q,k,v),['wait','launch','attention','save']),
                                           ('model.layers.0.self_attn.attn',None,['wait','launch','original_views','attention','save']),
                                           ('custom',(q,k,v),['wait','launch','original_views','attention','save'])]:
                events.clear();state.views=views;self.assertTrue(invoke(name));self.assertEqual(events,expected)
            events.clear();state.error=RuntimeError('submitted')
            with self.assertRaisesRegex(RuntimeError,'submitted'):invoke()
            self.assertEqual(events,['wait','launch'])
            events.clear();state.error=None;state.output=None
            with self.assertRaisesRegex(RuntimeError,'support changed'):invoke()
            self.assertEqual(events,['wait','launch'])


class DeviceContract(unittest.TestCase):
    def test_bf16_exact_views_aliases_and_sentinels(self):
        import torch
        ns = functions(torch)
        for row in (2560, 2601):
            for offset in (0, 37):
                storage = torch.full((offset+4*row+19,), -13., dtype=torch.bfloat16, device='cuda')
                x = storage.as_strided((4,2560), (row,1), offset)
                x.copy_(torch.arange(4*2560,device='cuda').reshape(4,2560).to(torch.bfloat16))
                before = storage.clone()
                a = ns['_original_qkv_views'](x)
                b = ns['_direct_qkv_views'](x)
                for old,new in zip(a,b):
                    self.assertTrue(torch.equal(old,new))
                    self.assertEqual(old.stride(),new.stride())
                    self.assertEqual(old.storage_offset(),new.storage_offset())
                    self.assertEqual(old.data_ptr(),new.data_ptr())
                self.assertTrue(torch.equal(storage,before))
                # Mutating each complete view reaches exactly its original columns.
                for view, begin, end in zip(b,(0,2048,2304),(2048,2304,2560)):
                    view.fill_(7.)
                    expected_storage=before.clone()
                    expected=expected_storage.as_strided((4,2560),(row,1),offset)
                    expected[:,begin:end].fill_(7.)
                    self.assertTrue(torch.equal(storage,expected_storage))
                    storage.copy_(before)
                torch.cuda.synchronize()


if __name__ == '__main__': unittest.main()
