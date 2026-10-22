# SPDX-License-Identifier: Apache-2.0
import gc,importlib,inspect,json,os
from pathlib import Path
import unittest
from unittest.mock import patch

class ActualDownLayout(unittest.TestCase):
 def test_real42_exact_reference_stream_and_invalidation(self):
    import torch as t
    from vllm import LLM,SamplingParams
    from c500_gpu_guard import GpuLease
    from c500_checkpoint_post_load import reclaim_after_load
    path=Path(os.environ['C500_FROZEN_PROTOCOL']);p=json.loads(path.read_text())
    output=Path(os.environ['C500_NUMERICAL_CAPTURE_PATH']).parent
    original=None;project=None;plugin=None;op=None;gate=None
    captured={};fallbacks=set();requests=[];llm=None;shutdown=False
    def observe(x,w,record):
      caller=inspect.currentframe().f_back
      layer=caller.f_locals['layer'];mlp=caller.f_locals['mlp']
      if layer in captured:return original(x,w,record)
      saved=x.clone();packed=record[0];packed_saved=packed.clone()
      result=original(x,w,record)
      self.assertIsNotNone(result,'actual object/layout unsupported')
      ref,_=mlp.down_proj(x)
      bad=result!=ref
      t.save(dict(layer=layer,x=x.cpu(),actual=result.cpu(),reference=ref.cpu(),coordinates=bad.nonzero().cpu(),weight_type=type(w).__module__+'.'+type(w).__qualname__,version=w._version,meta=gate._baseline_record(x,w)),output/f'actual-down-{layer:02d}.pt')
      self.assertTrue(t.equal(result,ref),'all real down layout violations retained')
      self.assertTrue(t.equal(saved,x));self.assertTrue(t.equal(packed,packed_saved))
      # Exact decoded BF16 products, independent FP64 sum; conservative FP32
      # accumulation bound and one BF16 rounding apply to both layouts.
      a=x.double().cpu();ww=w.double().cpu();dot=a@ww.T;absolute=a.abs()@ww.abs().T
      gamma=6144*2.0**-24/(1-6144*2.0**-24)
      acc=gamma*absolute;bound=acc+(dot.abs()+acc)/256+1e-7
      violation=(result.double().cpu()-dot).abs()>bound
      if violation.any():t.save(dict(coordinates=violation.nonzero(),actual=result.double().cpu()[violation],reference=dot[violation],bound=bound[violation]),output/f'down-reference-failure-{layer}.pt')
      self.assertFalse(bool(violation.any()))
      if layer==0:
        stream=t.cuda.Stream();stream.wait_stream(t.cuda.current_stream())
        with t.cuda.stream(stream):hot=original(x,w,record)
        stream.synchronize();self.assertTrue(t.equal(hot,ref))
        self.assertIsNone(original(x[:3],w,record))
        self.assertIsNone(original(x[:,::2],w,record))
        with t.enable_grad():self.assertIsNone(original(x,w,record))
      captured[layer]=dict(layer=layer,shape=list(x.shape),phase='decode',exact=True,independent_fp64=True)
      return result
    def observe_project(mlp,x):
      result=project(mlp,x)
      if x.shape[0] in (1,2,3):fallbacks.add(x.shape[0])
      return result
    try:
      with GpuLease() as lease:
        llm=LLM(model=p['model'],**p['engine'])
        plugin=importlib.import_module('vllm_fl.models.minicpm5_down_kn')
        op=importlib.import_module('flag_gems.runtime.backend._metax.fused.down_weight_kn')
        gate=importlib.import_module('flag_gems.runtime.backend._metax.fused.gate_up_silu')
        original=op.try_down_kn;project=plugin.project_down_kn

        reclaim_after_load(path.parent,output/'checkpoint-post-load.json',lease,t.cuda.synchronize,p['model'],p['checkpoint_post_load_reclaim'])
        runner=llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
        layers=runner.model.model.layers
        info=[]
        for i,layer in enumerate(layers):
          down=layer.mlp.down_proj;record=getattr(down,'_c500_kn_record',None)
          info.append(dict(layer=i,weight_type=type(down.weight).__qualname__,stamp_available=op.tensor_stamp(down.weight) is not None,packed=record is not None,bytes=record[0].numel()*record[0].element_size() if record else 0))
        (output/'post-load-layout.json').write_text(json.dumps(info,indent=2))
        self.assertTrue(all(r['packed'] for r in info),'untrackable weights must use original path; no model screening')
        self.assertEqual(sum(r['bytes'] for r in info),42*2048*6144*2)
        selected=[r for r in json.loads((path.parent/'main-selection.json').read_text()) if r['scenario']=='prefill-1k'][:4]
        with patch.object(op,'try_down_kn',observe),patch.object(plugin,'project_down_kn',observe_project):
          for m in (4,):
            values=llm.generate([dict(prompt_token_ids=r['prompt_token_ids']) for r in selected[:m]],SamplingParams(temperature=0,max_tokens=16,ignore_eos=True,seed=20261022),use_tqdm=False)
            self.assertEqual(len(values),m)
            for v in values:
              self.assertEqual(len(v.outputs[0].token_ids),16)
              requests.append(dict(batch=m,request_id=v.request_id,tokens=v.outputs[0].token_ids,prompt_tokens=len(v.prompt_token_ids)))
            (output/'real-model-requests.json').write_text(json.dumps(requests,indent=2))
        self.assertEqual(set(captured),set(range(42)));self.assertEqual(fallbacks,{1,2,3})
        with t.inference_mode():
          mlp=layers[0].mlp;down=mlp.down_proj;record=down._c500_kn_record
          x=t.zeros(4,6144,device='cuda',dtype=t.bfloat16)
          # A normal in-place update bumps the original version; no timed repack.
          down.weight.add_(0)
          self.assertIsNone(original(x,down.weight,record))
          down.quant_method.process_weights_after_loading(down)
          self.assertIsNotNone(original(x,down.weight,down._c500_kn_record))
          down._c500_kn_weight.add_(0)
          self.assertIsNone(original(x,down.weight,down._c500_kn_record))
          # Rebuild only through the explicit post-load boundary.
          down.quant_method.process_weights_after_loading(down)
          class Broken:
            def __getitem__(self,grid):
              def run(*a,**kw):raise RuntimeError('submitted')
              return run
          generic=importlib.import_module('flag_gems.ops.linear')
          meta=gate._baseline_record(x,down.weight)
          with patch.object(op,'_baseline_record',return_value=meta),patch.object(generic.linear_kernel,'jit_function',Broken()):
            with self.assertRaisesRegex(RuntimeError,'submitted'):original(x,down.weight,down._c500_kn_record)
    finally:
      if llm is not None:
        llm.llm_engine.engine_core.shutdown();shutdown=True;del llm;gc.collect()
      (output/'real-model-lifecycle.json').write_text(json.dumps(dict(source_refs=p['source_refs'],expected_requests=4,completed_requests=len(requests),captured=list(captured.values()),shutdown=shutdown,observer_restored=op is None or op.try_down_kn is original),indent=2))
