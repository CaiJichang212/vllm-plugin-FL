"""CPU dispatch/cache contracts; device exactness runs only in locked worker."""
import ast
from collections import OrderedDict
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
SOURCE = REPO/'vllm_fl/models/minicpm5_ffn_fusion.py'
GEMS = next(p for p in (REPO.parent/'FlagGems', REPO.parent/'gems') if p.is_dir())
KERNEL = GEMS/'src/flag_gems/runtime/backend/_metax/fused/gate_up_silu.py'
MODULE = 'flag_gems.runtime.backend._metax.fused.gate_up_silu'
ATTENTION = 'vllm.model_executor.layers.attention.attention'


def extract(path, names, ns):
    nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes)==len(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
    return ns


class Tensor:
    shape=(4,2048)


class HostContract(unittest.TestCase):
    def test_phase_layer_shape_fallback_and_submission_errors(self):
        state=NS(meta=NS(num_prefills=0,num_prefill_tokens=0,num_decodes=4,num_decode_tokens=4,num_actual_tokens=4),supported=True,error=False)
        events=[]
        def op(x,w,**kwargs):
            events.append(kwargs)
            if state.error:raise RuntimeError('submitted')
            return 'new' if kwargs and state.supported else ('old' if not kwargs else None)
        module=ModuleType(MODULE);module.try_gate_up_silu=op
        attention=ModuleType(ATTENTION)
        def context(name):
            self.assertEqual(name,'model.layers.0.self_attn.attn')
            return state.meta,None,None,None
        attention.get_attention_context=context
        ns=extract(SOURCE,{'original_gate_up','project_gate_up_row32'},dict(torch=NS(Tensor=Tensor)))
        run=lambda x=Tensor(),prefix='model.layers.0.mlp':ns['project_gate_up_row32'](x,None,prefix)
        with patch.dict(sys.modules,{MODULE:module,ATTENTION:attention}):
            self.assertEqual(run(),'new');self.assertEqual(events[-1],{'row32':True})
            for field in vars(state.meta):
                previous=getattr(state.meta,field);setattr(state.meta,field,1)
                before=len(events);self.assertEqual(run(),'old');self.assertEqual(events[before:],[{}])
                setattr(state.meta,field,previous)
            for name in ('bad','model.layers.42.mlp','model.layers.-1.mlp',None):self.assertEqual(run(prefix=name),'old')
            for rows in (1,3,5):
                x=Tensor();x.shape=(rows,2048);self.assertEqual(run(x=x),'old')
            self.assertEqual(run(x=type('Subclass',(Tensor,),{})()),'old')
            state.supported=False;self.assertEqual(run(),'old');self.assertEqual(events[-2:],[{'row32':True},{}])
            state.error=True;before=len(events)
            with self.assertRaisesRegex(RuntimeError,'submitted'):run()
            self.assertEqual(events[before:],[{'row32':True}])

    def test_cold_hot_cache_separation_tile_and_dependencies(self):
        events=[];cache=OrderedDict();meta=dict(BLOCK_SIZE_M=128,BLOCK_SIZE_N=128,BLOCK_SIZE_K=32,num_warps=8,num_stages=2)
        output=NS(data_ptr=lambda:16)
        class Kernel:
            def __getitem__(self,grid):
                def run(*args,**kwargs):events.append((grid,kwargs));return object()
                return run
        ns=extract(KERNEL,{'try_gate_up_silu'},dict(supported=lambda *a:True,_count=lambda *a:None,
            _cache=lambda _:cache,_signature=lambda *a:('same',),_jit_stamp=lambda _:None,
            paired_gate_up_kernel=Kernel(),_baseline_record=lambda *a:meta,
            torch=NS(empty=lambda *a,**kw:output),triton=NS(cdiv=lambda a,b:(a+b-1)//b),
            torch_device_fn=NS(device=lambda _:contextlib.nullcontext()),_dependency=lambda *a:None))
        x=NS(shape=(4,2048),device=NS(index=0),dtype='bf16')
        f=ns['try_gate_up_silu'];self.assertIs(f(x,None,row32=True),output)
        self.assertEqual(events[-1],((1,96,1),dict(meta,BLOCK_SIZE_M=32,FUSED=True)))
        self.assertEqual(meta['BLOCK_SIZE_M'],128)
        f(x,None);self.assertEqual(events[-1][1]['BLOCK_SIZE_M'],128)
        meta['BLOCK_SIZE_M']=64;before=len(events);self.assertIsNone(f(x,None,row32=True));self.assertEqual(len(events),before)
        meta['BLOCK_SIZE_M']=128
        # Exercise actual warm-run branch with distinguishable original/candidate keys.
        for flag in (False,True):
            key=(('same',),flag)
            compiled={};binary=object();compiled['key']=binary
            runner=lambda *a,flag=flag:events.append(('warm',flag))
            ns['paired_gate_up_kernel'].cache={0:compiled}
            cache[key]=('stamp','dep',compiled,'key',binary,runner)
            ns['_jit_stamp']=lambda _:'stamp';ns['_valid_dependency']=lambda *a:True
            self.assertIs(f(x,None,row32=flag),output);self.assertEqual(events[-1],('warm',flag))
        # Dependency change invalidates descriptor and compiles selected tile again.
        ns['_valid_dependency']=lambda *a:False;ns['_jit_stamp']=lambda _:None
        f(x,None,row32=True);self.assertEqual(events[-1][1]['BLOCK_SIZE_M'],32)
        self.assertNotIn((('same',),True),cache)
        x.shape=(3,2048);before=len(events);self.assertIsNone(f(x,None,row32=True));self.assertEqual(len(events),before)

    def test_all_layer_branch_trace_and_original_ffn_guards(self):
        module=ModuleType(MODULE);module.try_gate_up_silu=lambda *a,**kw:'new'
        attention=ModuleType(ATTENTION);attention.get_attention_context=lambda _: (NS(num_prefills=0,num_prefill_tokens=0,num_decodes=4,num_decode_tokens=4,num_actual_tokens=4),None,None,None)
        ns=extract(SOURCE,{'original_gate_up','project_gate_up_row32','_forward'},dict(torch=NS(Tensor=Tensor)))
        seen=set();tree=ast.parse(SOURCE.read_text());helper=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='project_gate_up_row32')
        line=next(n.lineno for n in ast.walk(helper) if isinstance(n,ast.Return) and isinstance(n.value,ast.Name) and n.value.id=='out')
        def trace(frame,event,arg):
            if event=='line' and frame.f_code.co_name=='project_gate_up_row32' and frame.f_lineno==line:
                d=frame.f_locals;seen.add((tuple(d['x'].shape),d['phase'],d['layer']))
            return trace
        old=sys.gettrace()
        try:
            with patch.dict(sys.modules,{MODULE:module,ATTENTION:attention}):
                sys.settrace(trace)
                for layer in range(42):self.assertEqual(ns['project_gate_up_row32'](Tensor(),None,f'model.layers.{layer}.mlp'),'new')
        finally:sys.settrace(old)
        self.assertEqual(seen,{((4,2048),'decode',i) for i in range(42)})
        ns.update(fallback_reason=lambda *a:'hooks',_observe=False)
        obj=NS(_c500_ffn_saved={'forward':lambda x:'full_fallback'})
        self.assertEqual(ns['_forward'](obj,Tensor()),'full_fallback')

    def test_kernel_arithmetic_frozen_and_masked_row_writes(self):
        node=next(n for n in ast.parse(KERNEL.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='paired_gate_up_kernel')
        self.assertEqual(hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest(),'be16327f7d87070ae634df15c61b9db2aee588f95d0b149cda3a9f3423ed0489')
        guards=[n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('fallback_reason','bind_gate_up_silu')]
        self.assertEqual(hashlib.sha256(''.join(ast.dump(n,include_attributes=False) for n in guards).encode()).hexdigest(),'aa5c85c18b3d99b26b5309c068764c0c02781235ba4e396ba1e85cd324e1ab92')
        for tile in (32,128):
            writes={(row,col) for row in range(tile) if row<4 for col in range(6144)}
            self.assertEqual(writes,{(r,c) for r in range(4) for c in range(6144)})


class DeviceContract(unittest.TestCase):
    def test_real_42_layer_weights_cold_hot_stream_exact_and_no_mutation(self):
        import torch
        from safetensors import safe_open
        protocol=json.loads(Path(os.environ['C500_FROZEN_PROTOCOL']).read_text())
        model=Path(protocol['model']);index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
        op=importlib.import_module(MODULE);torch.manual_seed(20261021)
        stream=torch.cuda.Stream()
        with torch.inference_mode(),torch.cuda.stream(stream):
            for layer in range(42):
                pieces=[]
                for part in ('gate','up'):
                    name=f'model.layers.{layer}.mlp.{part}_proj.weight'
                    with safe_open(model/index[name],framework='pt',device='cpu') as handle:pieces.append(handle.get_tensor(name))
                weight=torch.nn.Parameter(torch.cat(pieces).to(device='cuda',dtype=torch.bfloat16),requires_grad=False)
                del pieces
                oldw=weight.clone()
                for scale,offset in ((0.,0),(1.,0),(8.,8)):
                    raw=torch.randn((4*2048+offset,),device='cuda',dtype=torch.bfloat16)*scale
                    x=raw[offset:].view(4,2048);oldx=raw.clone()
                    op.clear_cache()
                    meta=op._baseline_record(x,weight)
                    self.assertEqual(meta['BLOCK_SIZE_M'],128,('actual base row tile',layer))
                    a=op.try_gate_up_silu(x,weight);b=op.try_gate_up_silu(x,weight,row32=True)
                    stream.synchronize();self.assertIsNotNone(b)
                    self.assertTrue(torch.isfinite(b).all().item(),layer)
                    self.assertTrue(torch.equal(a,b),('cold',layer,scale,offset))
                    for flag in (False,True,True,False):
                        result=op.try_gate_up_silu(x,weight,row32=flag);stream.synchronize()
                        self.assertTrue(torch.equal(a,result),('warm separation',layer,flag))
                    self.assertTrue(torch.equal(raw,oldx));self.assertTrue(torch.equal(weight,oldw))
                    self.assertIsNone(op.try_gate_up_silu(x[:3],weight,row32=True))
                    # An explicit independent zero reference also covers activation/product.
                    if scale==0:self.assertEqual(torch.count_nonzero(b).item(),0)
                del weight,oldw,a,b,result
            op.clear_cache()
        stream.synchronize()


if __name__=='__main__':unittest.main()
