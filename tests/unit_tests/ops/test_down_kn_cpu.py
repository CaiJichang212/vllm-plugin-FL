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
    self.assertIsNone(ns['try_down_kn'](x,self.w,(packed,old,ps)))
 def test_no_record_rejected(self):
    ns=extract({'try_down_kn'},dict(self.ns));self.assertIsNone(ns['try_down_kn'](None,None,None))
if __name__=='__main__':unittest.main(verbosity=2)
