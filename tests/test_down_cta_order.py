# SPDX-License-Identifier: Apache-2.0
"""CPU dispatch contracts and locked real-input down-projection validation."""
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

REPO=Path(__file__).resolve().parents[1]
SOURCE=REPO/'vllm_fl/models/minicpm5_ffn_fusion.py'
GEMS=next(p for p in (REPO.parent/'FlagGems',REPO.parent/'gems') if p.is_dir())
KERNEL=GEMS/'src/flag_gems/runtime/backend/_metax/fused/down_cta_order.py'
MODULE='flag_gems.runtime.backend._metax.fused.down_cta_order'
ATTENTION='vllm.model_executor.layers.attention.attention'


def extract(path,names,namespace):
    nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes)==len(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace


class Tensor:
    shape=(4,6144)
    ndim=2


class HostContract(unittest.TestCase):
    def test_real_branch_domain_and_post_submit_error(self):
        meta=NS(num_prefills=0,num_prefill_tokens=0,num_decodes=4,num_decode_tokens=4,num_actual_tokens=4)
        attention=ModuleType(ATTENTION);attention.get_attention_context=lambda _:(meta,None,None,None)
        op=ModuleType(MODULE);events=[]
        def kernel(*args):events.append('candidate');return 'candidate'
        op.try_down_cta_order=kernel
        class Down:
            weight=object()
            def __call__(self,x):events.append('fallback');return 'fallback',None
        mlp=NS(_c500_ffn_saved={'prefix':'model.layers.0.mlp'},down_proj=Down())
        ns=extract(SOURCE,{'project_down_cta_order'},dict(torch=NS(Tensor=Tensor)))
        with patch.dict(sys.modules,{MODULE:op,ATTENTION:attention}):
            for layer in range(42):
                mlp._c500_ffn_saved['prefix']=f'model.layers.{layer}.mlp'
                self.assertEqual(ns['project_down_cta_order'](mlp,Tensor()),'candidate')
            for m in (1,2,3,4):
                x=Tensor();x.shape=(m,6144)
                for field in ('num_decodes','num_decode_tokens','num_actual_tokens'):setattr(meta,field,m)
                self.assertEqual(ns['project_down_cta_order'](mlp,x),'candidate')
            for field in ('num_decodes','num_decode_tokens','num_actual_tokens'):setattr(meta,field,4)
            for name in vars(meta):
                old=getattr(meta,name);setattr(meta,name,1)
                self.assertEqual(ns['project_down_cta_order'](mlp,Tensor()),'fallback');setattr(meta,name,old)
            for prefix in ('bad','model.layers.42.mlp',None):
                mlp._c500_ffn_saved['prefix']=prefix;self.assertEqual(ns['project_down_cta_order'](mlp,Tensor()),'fallback')
            mlp._c500_ffn_saved['prefix']='model.layers.0.mlp'
            for m in (0,5,1024):
                x=Tensor();x.shape=(m,6144);self.assertEqual(ns['project_down_cta_order'](mlp,x),'fallback')
            op.try_down_cta_order=lambda *a:None
            self.assertEqual(ns['project_down_cta_order'](mlp,Tensor()),'fallback')
            def fail(*args):events.append('submitted');raise RuntimeError('submitted')
            op.try_down_cta_order=fail;before=len(events)
            with self.assertRaisesRegex(RuntimeError,'submitted'):ns['project_down_cta_order'](mlp,Tensor())
            self.assertEqual(events[before:],['submitted'])

    def test_original_module_guards_remain_and_run_first(self):
        guards=[n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('fallback_reason','bind_gate_up_silu')]
        self.assertEqual(hashlib.sha256(''.join(ast.dump(n,include_attributes=False) for n in guards).encode()).hexdigest(),'aa5c85c18b3d99b26b5309c068764c0c02781235ba4e396ba1e85cd324e1ab92')
        ns=extract(SOURCE,{'_forward'},dict(fallback_reason=lambda *a:'submodule_hooks',_observe=False))
        self.assertEqual(ns['_forward'](NS(_c500_ffn_saved={'forward':lambda x:'saved'}),Tensor()),'saved')

    def test_cold_hot_invalidation_and_suppressed_launch(self):
        cache=OrderedDict();events=[];binary=object();compiled={};meta=dict(BLOCK_SIZE_M=128,BLOCK_SIZE_N=32,BLOCK_SIZE_K=32,num_warps=4,num_stages=4)
        class Jit:
            debug=False
            cache={0:compiled}
            def binder(self,*args,**kw):return None,['sig'],{},None,{}
            def __getitem__(self,grid):
                def run(*args,**kw):
                    events.append(('launch',grid,kw));compiled['sig'+str(({},{}))]=binary;return binary
                return run
        class Binary:
            def __getitem__(self,grid):return lambda *args:events.append(('warm',grid))
        binary=Binary();jit=Jit();out=NS(data_ptr=lambda:16)
        dep=NS(valid=True);launch=NS(_jit_stamp=lambda _:'stamp',_signature=lambda *a:'sig',
            _valid_dependency=lambda *a:dep.valid,_baseline_record=lambda *a:meta,_dependency=lambda *a:'dep')
        ns=extract(KERNEL,{'try_down_cta_order'},dict(supported=lambda *a:True,launch=launch,_cache=lambda _:cache,
            down_cta_order_kernel=jit,torch=NS(empty=lambda *a,**kw:out),
            triton=NS(cdiv=lambda a,b:(a+b-1)//b),torch_device_fn=NS(device=lambda _:contextlib.nullcontext())))
        x=NS(device=NS(index=0),dtype='bf16',shape=(4,6144));run=lambda:ns['try_down_cta_order'](x,None)
        self.assertIs(run(),out);self.assertEqual(events[-1],('launch',(1,64,1),dict(meta,SWIZZLE=True)))
        self.assertIs(run(),out);self.assertEqual(events[-1][0],'warm')
        dep.valid=False;self.assertIs(run(),out);self.assertEqual(events[-1][0],'launch')
        cache.clear();meta['BLOCK_SIZE_K']=4096;before=len(events);self.assertIsNone(run());self.assertEqual(len(events),before)
        meta['BLOCK_SIZE_K']=32;launch._jit_stamp=lambda _:None
        self.assertIsNone(run());self.assertEqual(len(events),before)

    def test_true_return_site_is_distinct_from_fallback(self):
        fn=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='project_down_cta_order')
        guarded=[n for n in ast.walk(fn) if isinstance(n,ast.If) and ast.unparse(n.test)=='out is not None']
        self.assertEqual(len(guarded),1)
        candidate=guarded[0].body[0];fallback=fn.body[-1]
        self.assertIsInstance(candidate,ast.Return);self.assertEqual(ast.unparse(candidate.value),'out')
        self.assertIsInstance(fallback,ast.Return);self.assertNotEqual(candidate.lineno,fallback.lineno)

    def test_permutation_and_unchanged_reduction(self):
        tree=ast.parse(KERNEL.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='down_cta_order_kernel')
        loops=[n for n in ast.walk(fn) if isinstance(n,ast.For)];self.assertEqual(len(loops),1)
        self.assertEqual(ast.unparse(loops[0].iter),'range(0, tl.cdiv(K, BLOCK_SIZE_K))')
        dots=[n for n in ast.walk(fn) if isinstance(n,ast.Call) and ast.unparse(n.func)=='tl.dot'];self.assertEqual(len(dots),1)
        mapping=[(pid%8)*8+pid//8 for pid in range(64)]
        self.assertEqual(sorted(mapping),list(range(64)));self.assertNotEqual(mapping,list(range(64)))
        self.assertEqual([mapping[p] for p in mapping],list(range(64)))
        self.assertEqual(sorted(col for tile in mapping for col in range(tile*32,(tile+1)*32)),list(range(2048)))
        self.assertNotIn('cache_modifier',KERNEL.read_text())
        # Except for CTA-to-column mapping, the candidate/control compile the same body.
        assignment=next(n for n in fn.body if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='column_tile')
        self.assertIsInstance(assignment.value,ast.IfExp)
        self.assertEqual(ast.unparse(assignment.value.orelse),'pid')



class DeviceContract(unittest.TestCase):
    def test_actual42_exact_fp64_stream_offsets_alias_and_sentinels(self):
        import torch as t
        from safetensors import safe_open
        import vllm.model_executor.parameter as parameter
        p=json.loads(Path(os.environ['C500_FROZEN_PROTOCOL']).read_text())
        manifests=[]
        for name in p['declaration']['evidence']:
            if Path(name).name=='real-inputs.json':manifests.append(json.loads(Path(name).read_text()))
        self.assertEqual(len(manifests),1);manifest=manifests[0]
        self.assertEqual(manifest['kind'],'down-cta-order-inputs-v1');self.assertEqual(manifest['model'],p['model'])
        model=Path(p['model']);index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
        op=importlib.import_module(MODULE);linear=importlib.import_module('_metax.ops.linear')
        output=Path(os.environ['C500_NUMERICAL_CAPTURE_PATH']).parent;rows=[]
        t.manual_seed(20261023);stream=t.cuda.Stream()
        with t.inference_mode(),t.cuda.stream(stream):
            for item in manifest['rows']:
                layer=item['layer'];file=Path(item['path'])
                self.assertEqual(hashlib.sha256(file.read_bytes()).hexdigest(),item['sha256'])
                actual=t.load(file,map_location='cpu',weights_only=True);self.assertEqual(actual['layer'],layer)
                name=f'model.layers.{layer}.mlp.down_proj.weight'
                with safe_open(model/index[name],framework='pt',device='cpu') as handle:w=handle.get_tensor(name).to('cuda')
                with patch.object(parameter,'get_tensor_model_parallel_rank',return_value=0),patch.object(parameter,'get_tensor_model_parallel_world_size',return_value=1):
                    weight=parameter.ModelWeightParameter(data=w,input_dim=1,output_dim=0,weight_loader=lambda *a:None)
                savedw=weight.clone();original=actual['x'].to('cuda')
                cases=[(f'actual-m{m}',original[:m]) for m in (1,2,3,4)]+[(f'scale{scale}-m{m}',t.randn((m,6144),device='cuda',dtype=t.bfloat16)*scale) for m in (1,2,3,4) for scale in (0.,1.,8.)]
                for label,x in cases:
                    saved=x.clone();op.clear_cache();a=linear.linear(x,weight);meta=op.launch._baseline_record(x,weight)
                    self.assertEqual(meta,item['meta'])
                    b=op.try_down_cta_order(x,weight);stream.synchronize();self.assertIsNotNone(b)
                    if label.startswith('actual-m'):self.assertTrue(t.equal(a.cpu(),actual['reference'][:x.shape[0]]),'historical input baseline output changed')

                    if layer==0 and label.startswith('actual-m'):
                        record=next(iter(op._cache(x.device.index).values()));candidate_kernel=record[4]
                        control=t.empty((x.shape[0],2048),device=x.device,dtype=x.dtype)
                        control_kernel=op.down_cta_order_kernel[(1,64,1)](x,weight,control,x.shape[0],6144,2048,SWIZZLE=False,**meta)
                        stream.synchronize();self.assertTrue(t.equal(a,control))
                        binaries={}
                        for tag,kernel in [('linear_order',control_kernel),('transposed_order',candidate_kernel)]:
                            binary=kernel.asm['mcfatbin']
                            self.assertIsInstance(binary,bytes)
                            (output/f'cta-m{x.shape[0]}-{tag}.mcfatbin').write_bytes(binary)
                            binaries[tag]=dict(bytes=len(binary),sha256=hashlib.sha256(binary).hexdigest(),asm_keys=list(kernel.asm))
                        changed=binaries['linear_order']['sha256']!=binaries['transposed_order']['sha256']
                        (output/f'cta-order-codegen-m{x.shape[0]}.json').write_text(json.dumps(dict(binaries=binaries,different=changed,scope='machine-code identity differs; arithmetic equality separately checked; does not establish hardware issue order or benefit'),indent=2))
                        self.assertTrue(changed,'compiler erased the candidate mechanism; no model screen')

                    bad=a!=b
                    if bad.any():t.save(dict(layer=layer,label=label,x=x.cpu(),A=a.cpu(),B=b.cpu(),coordinates=bad.nonzero().cpu(),meta=meta),output/f'exact-failure-{layer}-{label}.pt')
                    self.assertTrue(t.equal(a,b),(layer,label,'exact'))
                    for _ in range(2):self.assertTrue(t.equal(a,op.try_down_cta_order(x,weight)))
                    aa=x.double().cpu();ww=weight.double().cpu();reference=aa@ww.T;magnitude=aa.abs()@ww.abs().T
                    gamma=6144*2.0**-24/(1-6144*2.0**-24);error=gamma*magnitude;bound=error+(reference.abs()+error)/256+1e-7
                    violation=(b.double().cpu()-reference).abs()>bound
                    if violation.any():t.save(dict(layer=layer,label=label,coordinates=violation.nonzero(),actual=b.double().cpu()[violation],reference=reference[violation],bound=bound[violation]),output/f'fp64-failure-{layer}-{label}.pt')
                    self.assertFalse(bool(violation.any()),(layer,label,'independent FP64'))
                    self.assertTrue(t.equal(saved,x));self.assertTrue(t.equal(savedw,weight))
                    del aa,ww,reference,magnitude,error,bound,violation
                if layer==0:
                    x=original;a=linear.linear(x,weight)
                    backing=t.full((4*2048+32,),-17,device='cuda',dtype=t.bfloat16);target=backing[16:-16].view(4,2048);allocate=t.empty
                    with patch.object(t,'empty',side_effect=lambda *args,**kw:target if args and args[0]==(4,2048) else allocate(*args,**kw)):
                        self.assertTrue(t.equal(a,op.try_down_cta_order(x,weight)))
                    self.assertTrue(bool((backing[:16]==-17).all()));self.assertTrue(bool((backing[-16:]==-17).all()))
                    offset=t.empty((x.numel()+8,),device='cuda',dtype=t.bfloat16)[8:].view_as(x);offset.copy_(x)
                    self.assertTrue(t.equal(a,op.try_down_cta_order(offset,weight)))
                    noncontiguous=t.empty((4,12288),device='cuda',dtype=t.bfloat16)[:,::2]
                    self.assertIsNone(op.try_down_cta_order(noncontiguous,weight));self.assertTrue(t.equal(linear.linear(x[:3],weight),op.try_down_cta_order(x[:3],weight)))
                    alias=weight.detach()[:4];self.assertTrue(t.equal(linear.linear(alias,weight),op.try_down_cta_order(alias,weight)));self.assertTrue(t.equal(savedw,weight))
                    with t.enable_grad():self.assertIsNone(op.try_down_cta_order(x,weight))
                    old_stamp=op.launch._jit_stamp
                    with patch.object(op.launch,'_jit_stamp',return_value=None):self.assertIsNone(op.try_down_cta_order(x,weight))
                    with patch.object(op.launch,'_valid_dependency',return_value=False):self.assertTrue(t.equal(a,op.try_down_cta_order(x,weight)))
                    self.assertIs(op.launch._jit_stamp,old_stamp)
                rows.append(dict(layer=layer,shapes=[[m,6144] for m in (1,2,3,4)],weight_type=type(weight).__qualname__,meta=meta,exact=True,fp64=True,cases=16))
                (output/'down-cta-order-coverage.json').write_text(json.dumps(rows,indent=2))
                del weight,w,savedw,cases,original,a,b,x
            op.clear_cache()
        stream.synchronize();self.assertEqual(len(rows),42)


if __name__=='__main__':unittest.main()
