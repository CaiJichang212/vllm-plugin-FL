# SPDX-License-Identifier: Apache-2.0
"""Guarded row-dispatch specialization; CPU contracts and real-weight checks."""
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

SOURCE=Path(__file__).resolve().parents[1]/'vllm_fl/models/minicpm5_ffn_fusion.py'
ATTENTION='vllm.model_executor.layers.attention.attention'


def extract(names,namespace):
    nodes=[n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes)==len(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE),'exec'),namespace)
    return namespace


class Tensor:
    shape=(4,6144)
    ndim=2


def context():
    meta=NS(num_prefills=0,num_prefill_tokens=0,num_decodes=4,num_decode_tokens=4,num_actual_tokens=4)
    module=ModuleType(ATTENTION);module.get_attention_context=lambda _:(meta,None,None,None)
    return meta,module


class HostContract(unittest.TestCase):
    def test_domain_fallback_and_error_propagation(self):
        events=[];meta,module=context()
        class Down:
            input_is_parallel=True
            return_bias=True
            quant_method=NS(apply=lambda *a:events.append('apply') or 'direct')
            def __call__(self,x):events.append('module');return 'fallback',None
        down=Down();mlp=NS(down_proj=down,_c500_ffn_saved={'prefix':'model.layers.0.mlp'})
        ns=extract({'project_down_direct'},dict(torch=NS(Tensor=Tensor)))
        with patch.dict(sys.modules,{ATTENTION:module}):
            for layer in range(42):
                mlp._c500_ffn_saved['prefix']=f'model.layers.{layer}.mlp'
                self.assertEqual(ns['project_down_direct'](mlp,Tensor()),'direct')
            for m in (1,2,3,4):
                x=Tensor();x.shape=(m,6144)
                for field in ('num_decodes','num_decode_tokens','num_actual_tokens'):setattr(meta,field,m)
                self.assertEqual(ns['project_down_direct'](mlp,x),'direct')
            for field in ('num_decodes','num_decode_tokens','num_actual_tokens'):setattr(meta,field,4)
            for field in vars(meta):
                old=getattr(meta,field);setattr(meta,field,1)
                self.assertEqual(ns['project_down_direct'](mlp,Tensor()),'fallback');setattr(meta,field,old)
            down._compiled_call_impl=object();self.assertEqual(ns['project_down_direct'](mlp,Tensor()),'fallback');down._compiled_call_impl=None
            for field in ('input_is_parallel','return_bias'):
                setattr(down,field,False);self.assertEqual(ns['project_down_direct'](mlp,Tensor()),'fallback');setattr(down,field,True)
            for m in (0,5,1024):
                x=Tensor();x.shape=(m,6144);self.assertEqual(ns['project_down_direct'](mlp,x),'fallback')
            for prefix in ('bad','model.layers.42.mlp',None):
                mlp._c500_ffn_saved['prefix']=prefix;self.assertEqual(ns['project_down_direct'](mlp,Tensor()),'fallback')
            mlp._c500_ffn_saved['prefix']='model.layers.0.mlp'
            def fail(*args):events.append('submitted');raise RuntimeError('submitted')
            down.quant_method.apply=fail;before=len(events)
            with self.assertRaisesRegex(RuntimeError,'submitted'):ns['project_down_direct'](mlp,Tensor())
            self.assertEqual(events[before:],['submitted'])

    def test_original_guards_are_unchanged_and_run_before_direct_path(self):
        guards=[n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('fallback_reason','bind_gate_up_silu')]
        self.assertEqual(hashlib.sha256(''.join(ast.dump(n,include_attributes=False) for n in guards).encode()).hexdigest(),'aa5c85c18b3d99b26b5309c068764c0c02781235ba4e396ba1e85cd324e1ab92')
        ns=extract({'_forward'},dict(fallback_reason=lambda *a:'submodule_hooks',_observe=False))
        self.assertEqual(ns['_forward'](NS(_c500_ffn_saved={'forward':lambda x:'saved'}),Tensor()),'saved')


class DeviceContract(unittest.TestCase):
    def test_real42_row_module_exact_fp64_and_current_dispatch(self):
        import torch as t
        from safetensors import safe_open
        from vllm.model_executor.layers.linear import RowParallelLinear, UnquantizedLinearMethod
        from vllm.model_executor.parameter import ModelWeightParameter
        p=json.loads(Path(os.environ['C500_FROZEN_PROTOCOL']).read_text())
        manifests=[json.loads(Path(name).read_text()) for name in p['declaration']['evidence'] if Path(name).name=='real-inputs.json']
        self.assertEqual(len(manifests),1);manifest=manifests[0];self.assertEqual(manifest['model'],p['model'])
        model=Path(p['model']);index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
        ns=extract({'project_down_direct'},dict(torch=t));meta,module=context();rows=[]
        output=Path(os.environ['C500_NUMERICAL_CAPTURE_PATH']).parent
        t.manual_seed(20261023);stream=t.cuda.Stream()
        with t.inference_mode(),t.cuda.stream(stream),patch.dict(sys.modules,{ATTENTION:module}):
            for item in manifest['rows']:
                layer=item['layer'];file=Path(item['path']);self.assertEqual(hashlib.sha256(file.read_bytes()).hexdigest(),item['sha256'])
                actual=t.load(file,map_location='cpu',weights_only=True);self.assertEqual(actual['layer'],layer)
                name=f'model.layers.{layer}.mlp.down_proj.weight'
                with safe_open(model/index[name],framework='pt',device='cpu') as handle:w=handle.get_tensor(name).to('cuda')
                # Real framework class and forward implementation, loaded real parameter.
                down=RowParallelLinear.__new__(RowParallelLinear);t.nn.Module.__init__(down)
                down.input_is_parallel=True;down.return_bias=True;down.tp_rank=0;down.tp_size=1
                down.skip_bias_add=False;down.reduce_results=True;down.bias=None
                down.quant_method=UnquantizedLinearMethod()
                import vllm.model_executor.parameter as parameter
                with patch.object(parameter,'get_tensor_model_parallel_rank',return_value=0),patch.object(parameter,'get_tensor_model_parallel_world_size',return_value=1):
                    down.weight=ModelWeightParameter(data=w,input_dim=1,output_dim=0,weight_loader=lambda *a:None)
                mlp=NS(down_proj=down,_c500_ffn_saved={'prefix':f'model.layers.{layer}.mlp'})
                original=actual['x'].to('cuda');savedw=w.clone()
                cases=[(f'actual-m{m}',original[:m]) for m in (1,2,3,4)]+[(f'scale{scale}-m{m}',t.randn((m,6144),device='cuda',dtype=t.bfloat16)*scale) for m in (1,2,3,4) for scale in (0.,1.,8.)]
                for label,x in cases:
                    for field in ('num_decodes','num_decode_tokens','num_actual_tokens'):setattr(meta,field,x.shape[0])
                    saved=x.clone();a,_=down(x);b=ns['project_down_direct'](mlp,x);stream.synchronize()
                    if not t.equal(a,b):t.save(dict(layer=layer,label=label,A=a.cpu(),B=b.cpu(),coordinates=(a!=b).nonzero().cpu()),output/f'exact-failure-{layer}-{label}.pt')
                    self.assertTrue(t.equal(a,b),(layer,label))
                    self.assertTrue(t.equal(a,ns['project_down_direct'](mlp,x)))
                    aa=x.double().cpu();ww=w.double().cpu();ref=aa@ww.T;mag=aa.abs()@ww.abs().T
                    gamma=6144*2**-24/(1-6144*2**-24);error=gamma*mag;bound=error+(ref.abs()+error)/256+1e-7
                    violation=(b.double().cpu()-ref).abs()>bound
                    if violation.any():t.save(dict(layer=layer,label=label,coordinates=violation.nonzero(),actual=b.double().cpu()[violation],reference=ref[violation],bound=bound[violation]),output/f'fp64-failure-{layer}-{label}.pt')
                    self.assertFalse(bool(violation.any()),(layer,label,'FP64'))
                    self.assertTrue(t.equal(x,saved));self.assertTrue(t.equal(w,savedw))
                    del aa,ww,ref,mag,error,bound,violation
                for field in ('num_decodes','num_decode_tokens','num_actual_tokens'):setattr(meta,field,4)
                if layer==0:
                    offset=t.empty((4*6144+8,),device='cuda',dtype=t.bfloat16)[8:].view(4,6144);offset.copy_(original)
                    for x in (offset,t.empty((4,12288),device='cuda',dtype=t.bfloat16)[:,::2].copy_(original),w[:4]):
                        a,_=down(x);self.assertTrue(t.equal(a,ns['project_down_direct'](mlp,x)))
                    # The direct path must use a changed apply implementation now, not a cached callable.
                    with patch.object(down.quant_method,'apply',return_value='current'):
                        self.assertEqual(ns['project_down_direct'](mlp,original),'current')
                rows.append(dict(layer=layer,shapes=[[m,6144] for m in (1,2,3,4)],exact=True,fp64=True,cases=16,weight_type=type(down.weight).__qualname__))
                (output/'down-direct-coverage.json').write_text(json.dumps(rows,indent=2))
                del down,mlp,w,savedw,original,cases,a,b,x
        stream.synchronize();self.assertEqual(len(rows),42)




class FrozenHitContract(unittest.TestCase):
    def test_frozen_product_and_branch_match_implemented_decode_domain(self):
        protocol = Path(os.environ['C500_FROZEN_PROTOCOL'])
        spec = json.loads(protocol.read_text())['declaration']['hit']
        self.assertEqual(spec['required_shapes'], [[m, 6144] for m in (1, 2, 3, 4)])
        self.assertEqual(spec['required_phases'], ['decode'])
        self.assertEqual(spec['required_layers'], list(range(42)))
        helper = next(n for n in ast.parse(SOURCE.read_text()).body
                      if isinstance(n, ast.FunctionDef) and n.name == spec['branch']['qualname'])
        guarded_returns = [node.lineno for stmt in helper.body if isinstance(stmt, ast.If)
                           for node in ast.walk(stmt) if isinstance(node, ast.Return)]
        self.assertIn(spec['branch']['line'], guarded_returns)

if __name__=='__main__':unittest.main()
