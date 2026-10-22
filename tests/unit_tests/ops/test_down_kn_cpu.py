# SPDX-License-Identifier: Apache-2.0
import ast,contextlib
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[3]
GEMS=next(p for p in (ROOT.parent/'FlagGems',ROOT.parent/'gems') if p.is_dir())
SOURCE=GEMS/'src/flag_gems/runtime/backend/_metax/fused/down_weight_kn.py'

def extract(names,ns):
    nodes=[n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE),'exec'),ns)
    return ns

class Tensor:
    _version=0;shape=(2048,6144);dtype='bf16';device=NS(type='cuda');requires_grad=False
    pointer=16
    def data_ptr(self):return self.pointer
    def stride(self):return (6144,1)
    def is_contiguous(self):return True

class Contracts(unittest.TestCase):
 def setUp(self):
    self.torch=NS(Tensor=Tensor,nn=NS(Parameter=Tensor))
    self.ns=extract({'tensor_stamp'},dict(torch=self.torch));self.w=Tensor()
 def test_normal_stamp_and_mutation(self):
    old=self.ns['tensor_stamp'](self.w);self.w._version+=1;self.assertNotEqual(old,self.ns['tensor_stamp'](self.w))
 def test_replacement(self):self.assertNotEqual(self.ns['tensor_stamp'](self.w),self.ns['tensor_stamp'](Tensor()))
 def test_pointer_change(self):
    old=self.ns['tensor_stamp'](self.w);self.w.pointer=32;self.assertNotEqual(old,self.ns['tensor_stamp'](self.w))
 def test_stride(self):
    old=self.ns['tensor_stamp'](self.w);self.w.stride=lambda:(1,2048);self.assertNotEqual(old,self.ns['tensor_stamp'](self.w))
 def test_unknown_type(self):self.assertIsNone(self.ns['tensor_stamp'](object()))
 def test_real_parameter_capability_subclass(self):self.assertIsNotNone(self.ns['tensor_stamp'](type('Parameter',(Tensor,),{})()))
 def test_inference_no_counter(self):
    class Inference(Tensor):
      @property
      def _version(self):raise RuntimeError('Inference tensors do not track version counter')
    self.assertIsNone(self.ns['tensor_stamp'](Inference()))
 def test_device_change(self):
    old=self.ns['tensor_stamp'](self.w);self.w.device=NS(type='cpu');self.assertNotEqual(old,self.ns['tensor_stamp'](self.w))
 def test_packed_mutation_rejected_before_launch(self):
    x=Tensor();x.shape=(4,6144);packed=Tensor();old=self.ns['tensor_stamp'](self.w);ps=self.ns['tensor_stamp'](packed);packed._version=2
    ns=extract({'try_down_kn'},dict(self.ns,_baseline_record=lambda *a:self.fail('launch preparation reached')))
    self.assertIsNone(ns['try_down_kn'](x,self.w,[packed,old,ps,True]))
 def test_no_record_rejected(self):
    ns=extract({'try_down_kn'},dict(self.ns));self.assertIsNone(ns['try_down_kn'](None,None,None))

class PhaseContract(unittest.TestCase):
 def test_stage_guard_and_no_post_submission_retry(self):
    import sys
    from types import ModuleType
    from unittest.mock import patch
    meta=NS(num_prefills=0,num_prefill_tokens=0,num_decodes=4,num_decode_tokens=4,num_actual_tokens=4)
    calls=[]
    attention=ModuleType('vllm.model_executor.layers.attention.attention')
    attention.get_attention_context=lambda prefix:(meta,None,None,None)
    op=ModuleType('flag_gems.runtime.backend._metax.fused.down_weight_kn')
    def candidate(*a):calls.append('new');return 'new'
    op.try_down_kn=candidate
    class Down:
      weight=NS(weight_loader='loader');_c500_kn_loader='loader';_c500_kn_record=[object(),None,None,True]
      def __init__(self):
        self._c500_kn_weight=self._c500_kn_record[0]
        self._c500_kn_apply=object()
        self.quant_method=NS(process_weights_after_loading='post',apply=self._c500_kn_apply)
        self._c500_kn_post_load='post'
      def __call__(self,x):calls.append('old');return 'old',None
    mlp=NS(down_proj=Down(),_c500_ffn_saved=dict(prefix='model.layers.0.mlp'))
    path=ROOT/'vllm_fl/models/minicpm5_down_kn.py'
    nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='project_down_kn']
    ns=dict(torch=NS(Tensor=Tensor),fallback_reason=lambda *a:None,get_attention_context=attention.get_attention_context)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
    x=Tensor();x.shape=(4,6144)
    with patch.dict(sys.modules,{op.__name__:op}):
      self.assertEqual(ns['project_down_kn'](mlp,x),'new')
      for field in vars(meta):
        old=getattr(meta,field);setattr(meta,field,1)
        self.assertEqual(ns['project_down_kn'](mlp,x),'old');setattr(meta,field,old)
      for prefix in ('bad',None,'model.layers.42.mlp'):
        mlp._c500_ffn_saved['prefix']=prefix;self.assertEqual(ns['project_down_kn'](mlp,x),'old')
      mlp._c500_ffn_saved['prefix']='model.layers.0.mlp'
      def broken(*a):calls.append('error');raise RuntimeError('submitted')
      op.try_down_kn=broken;start=len(calls)
      with self.assertRaisesRegex(RuntimeError,'submitted'):ns['project_down_kn'](mlp,x)
      self.assertEqual(calls[start:],['error'])

if __name__=='__main__':unittest.main(verbosity=2)
