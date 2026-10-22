# SPDX-License-Identifier: Apache-2.0
import importlib,json,os
from pathlib import Path
import unittest
from unittest.mock import patch

class RealModelTests(unittest.TestCase):
    def test_real_inputs_42_layers_and_fallback(self):
        import ast, gc, inspect, time
        import torch as t
        from vllm import LLM, SamplingParams
        from c500_gpu_guard import GpuLease
        from c500_checkpoint_post_load import reclaim_after_load
        plugin=importlib.import_module('vllm_fl.models.minicpm5_post_norm_gate')
        op=importlib.import_module('flag_gems.runtime.backend._metax.fused.post_norm_gate')
        gate=importlib.import_module('flag_gems.runtime.backend._metax.fused.gate_up_silu')
        path=Path(os.environ['C500_FROZEN_PROTOCOL']);p=json.loads(path.read_text())
        output=Path(os.environ['C500_NUMERICAL_CAPTURE_PATH']).parent
        captured={};fallbacks=set();requests=[];llm=None;shutdown=False
        original=op.try_post_norm_gate;project=plugin.project_post_norm
        def observed(x,r,nw,w,eps):
            frame=inspect.currentframe().f_back
            layer=frame.f_locals['layer'];norm=frame.f_locals['norm']
            if layer in captured:return original(x,r,nw,w,eps)
            before=[a.clone() for a in (x,r)]
            result=original(x,r,nw,w,eps)
            self.assertIsNotNone(result,'actual candidate silently fell back')
            xx,rr=norm(before[0].clone(),before[1].clone())
            ref=gate.try_gate_up_silu(xx,w)
            bad=result[0]!=ref
            sample=dict(layer=layer,x=before[0].cpu(),residual=before[1].cpu(),norm_weight=nw.cpu(),normalized=xx.cpu(),actual=result[0].cpu(),reference=ref.cpu(),coordinates=bad.nonzero().cpu(),meta=gate._baseline_record(xx,w))
            t.save(sample,output/f'actual-layer-{layer:02d}.pt')
            self.assertTrue(t.equal(result[0],ref),'real exact failure saved')
            self.assertTrue(t.equal(result[1],rr))
            self.assertTrue(all(t.equal(a,b) for a,b in zip(before,(x,r))))
            z=before[0].double()+before[1].double()
            expected=z/((z*z/2048).sum(-1,keepdim=True)+eps).sqrt()*nw.double()
            violations=(xx.double()-expected).abs()>1e-4+.016*expected.abs()
            if violations.any():t.save(dict(coordinates=violations.nonzero(),actual=xx[violations],reference=expected[violations]),output/f'actual-norm-failure-{layer}.pt')
            self.assertFalse(bool(violations.any()))
            captured[layer]=dict(layer=layer,shape=list(x.shape),phase='decode',exact=True,norm_reference=True,meta=sample['meta'])
            return result
        def observed_project(layer,x,r):
            result=project(layer,x,r)
            if x.shape[0] in (1,2,3):
                self.assertIsNone(result);fallbacks.add(x.shape[0])
            return result
        try:
            with GpuLease() as lease:
                llm=LLM(model=p['model'],**p['engine'])
                reclaim_after_load(path.parent,output/'real-checkpoint-post-load.json',lease,t.cuda.synchronize,p['model'],p['checkpoint_post_load_reclaim'])
                main=json.loads((path.parent/'main-selection.json').read_text())
                selected=[r for r in main if r['scenario']=='prefill-1k'][:4]
                with patch.object(op,'try_post_norm_gate',observed),patch.object(plugin,'project_post_norm',observed_project):
                    for m in (1,2,3,4):
                        values=llm.generate([dict(prompt_token_ids=r['prompt_token_ids']) for r in selected[:m]],SamplingParams(temperature=0,max_tokens=4,ignore_eos=True,seed=20261022),use_tqdm=False)
                        self.assertEqual(len(values),m)
                        for v in values:
                            self.assertEqual(len(v.outputs[0].token_ids),4)
                            requests.append(dict(batch=m,request_id=v.request_id,tokens=v.outputs[0].token_ids,prompt_tokens=len(v.prompt_token_ids)))
                        (output/'real-model-requests.json').write_text(json.dumps(requests,indent=2))
                self.assertEqual(set(captured),set(range(42)))
                self.assertEqual(fallbacks,{1,2,3})
                runner=llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
                layer=runner.model.model.layers[0]
                x=t.empty(4,2048,device='cuda',dtype=t.bfloat16)
                handle=layer.post_attention_layernorm.register_forward_hook(lambda *a:None)
                try:self.assertFalse(plugin.allowed(layer,x))
                finally:handle.remove()
                old=layer.post_attention_layernorm
                layer.post_attention_layernorm=t.nn.Identity()
                try:self.assertFalse(plugin.allowed(layer,x))
                finally:layer.post_attention_layernorm=old
        finally:
            if llm is not None:
                llm.llm_engine.engine_core.shutdown();shutdown=True;del llm;gc.collect()
            (output/'real-model-lifecycle.json').write_text(json.dumps(dict(source_refs=p['source_refs'],expected_requests=10,completed_requests=len(requests),captured=list(captured.values()),shutdown=shutdown,observer_restored=op.try_post_norm_gate is original),indent=2))
