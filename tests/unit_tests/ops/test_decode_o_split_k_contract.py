"""AST-isolated actual dispatch guard, no vendor imports or GPU work."""
import ast
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[3]/'vllm_fl/models/minicpm5_rope_kv.py'


class Unquantized: pass
class RowParallel:
    def __init__(self):
        self.quant_method=Unquantized(); self.tp_size=1; self.bias=None
        self.weight=object(); self.calls=0; self.hooked=False
        self.forward=lambda x: ('original',None)
    def __call__(self,x): self.calls+=1;return self.forward(x)


def build():
    tree=ast.parse(SOURCE.read_text())
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('project_decode_o','original_decode_o')]
    def module(name,**attrs):
        m=ModuleType(name);m.__dict__.update(attrs);return m
    hooks=NS()
    calls=[]
    def kernel(x,w,**kw):calls.append((x,w,kw));return 'candidate'
    mods={
        'vllm_fl.models.minicpm5_ffn_fusion':module('fusion',_hooked=lambda p:p.hooked),
        'torch.nn.modules':module('torch.nn.modules',module=hooks),
        'vllm.model_executor.layers.linear':module('linear',RowParallelLinear=RowParallel,UnquantizedLinearMethod=Unquantized),
        'flag_gems.runtime.backend._metax.fused.decode_o_split_k':module('kernel',try_decode_o=kernel)}
    ns={'__package__':'vllm_fl.models', '_dispatch_identity':lambda p:(id(p),p.forward,p.quant_method)}
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(SOURCE),'exec'),ns)
    proj=RowParallel();owner=NS(o_proj=proj,_decode_o_saved=ns['_dispatch_identity'](proj))
    return ns,owner,mods,hooks,calls


class DispatchContracts(unittest.TestCase):
    def test_candidate_and_current_arguments(self):
        ns,owner,mods,hooks,calls=build()
        with patch.dict(sys.modules,mods):
            x=object()
            self.assertEqual(ns['project_decode_o'](owner,x,'decode',41),'candidate')
            self.assertEqual(calls,[(x,owner.o_proj.weight,dict(phase='decode',layer=41))])
            self.assertEqual(owner.o_proj.calls,0)

    def test_hooks_dispatch_changes_bias_tp_adapters_and_replacement_fallback(self):
        for change in ('hook','global','bias','tp','adapter','forward','quant','replace','unbound'):
            ns,owner,mods,hooks,calls=build()
            if change=='hook':owner.o_proj.hooked=True
            if change=='global':hooks._global_forward_pre_hooks={0:object()}
            if change=='bias':owner.o_proj.bias=object()
            if change=='tp':owner.o_proj.tp_size=2
            if change=='adapter':owner.o_proj.adapter=object()
            if change=='forward':owner.o_proj.forward=lambda x:('changed',None)
            if change=='quant':owner.o_proj.quant_method=Unquantized()
            if change=='replace':owner.o_proj=RowParallel()
            if change=='unbound':del owner._decode_o_saved
            with patch.dict(sys.modules,mods):
                expected='changed' if change=='forward' else 'original'
                self.assertEqual(ns['project_decode_o'](owner,object(),'decode',0),expected,change)
            self.assertEqual(calls,[],change)
            self.assertEqual(owner.o_proj.calls,1,change)

    def test_unsupported_kernel_falls_back_once_and_submitted_error_never_retries(self):
        ns,owner,mods,hooks,calls=build()
        op=mods['flag_gems.runtime.backend._metax.fused.decode_o_split_k']
        with patch.dict(sys.modules,mods):
            op.try_decode_o=lambda *a,**kw:None
            self.assertEqual(ns['project_decode_o'](owner,object(),'decode',0),'original')
            self.assertEqual(owner.o_proj.calls,1)
            def fail(*a,**kw):raise RuntimeError('submitted')
            op.try_decode_o=fail
            with self.assertRaisesRegex(RuntimeError,'submitted'):
                ns['project_decode_o'](owner,object(),'decode',0)
            self.assertEqual(owner.o_proj.calls,1)


if __name__=='__main__':unittest.main()
