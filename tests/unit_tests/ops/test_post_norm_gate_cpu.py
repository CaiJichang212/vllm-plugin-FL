import ast, unittest
from pathlib import Path
from types import SimpleNamespace as N
from unittest.mock import Mock
P=Path(__file__).resolve().parents[3].parent
class T:
 shape=(4,2048);device='cuda';dtype='bf16';requires_grad=False;ptr=16;size=8192
 def data_ptr(self):return self.ptr
 def numel(self):return self.size
 def element_size(self):return 2
 def is_contiguous(self):return True
class Tests(unittest.TestCase):
 def setUp(self):
  import math
  source=P/'FlagGems/src/flag_gems/runtime/backend/_metax/fused/post_norm_gate.py'
  nodes=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='eligible']
  self.ns=dict(math=math,supported=lambda *a:True,torch=N(Tensor=T,nn=N(Parameter=T),is_grad_enabled=lambda:False))
  exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),self.ns)
  self.x=T();self.r=T();self.r.ptr=20000;self.nw=T();self.nw.ptr=40000;self.nw.shape=(2048,);self.nw.size=2048;self.w=T();self.w.ptr=50000
 def run_case(self):return self.ns['eligible'](self.x,self.r,self.nw,self.w,1e-5)
 def test_valid_disjoint(self):self.assertTrue(self.run_case())
 def test_alias_rejected(self):self.r.ptr=self.x.ptr;self.assertFalse(self.run_case())
 def test_adjacent_allowed(self):self.r.ptr=self.x.ptr+16384;self.assertTrue(self.run_case())
 def test_offset_misaligned(self):self.x.ptr=17;self.assertFalse(self.run_case())
 def test_empty(self):self.x.shape=(0,2048);self.assertFalse(self.run_case())
 def test_stride(self):self.r.is_contiguous=lambda:False;self.assertFalse(self.run_case())
 def test_dtype(self):self.nw.dtype='fp32';self.assertFalse(self.run_case())
 def test_parameter_subclass(self):
  self.nw=type('Parameter',(T,),{} )();self.nw.ptr=40000;self.nw.size=2048;self.nw.shape=(2048,);self.assertTrue(self.run_case())
 def test_nan_eps(self):self.assertFalse(self.ns['eligible'](self.x,self.r,self.nw,self.w,float('nan')))

class DispatchTests(unittest.TestCase):
 def setUp(self):
  import weakref
  from types import SimpleNamespace as NS
  self.NS=NS
  self.meta=NS(num_prefills=0,num_prefill_tokens=0,num_decodes=4,num_decode_tokens=4,num_actual_tokens=4)
  self.parent=NS(aux_hidden_state_layers=())
  norm=NS(variance_size_override=None,has_weight=True,hidden_size=2048)
  self.layer=NS(mlp=NS(),post_attention_layernorm=norm,_post_norm_parent=lambda:self.parent)
  modules=(self.layer.mlp,norm)
  self.layer._post_norm_saved=(modules,tuple(map(id,modules)),'model.layers.0')
  source=P/'vllm-plugin-FL/vllm_fl/models/minicpm5_post_norm_gate.py'
  nodes=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='allowed']
  self.ns=dict(torch=NS(Tensor=T),_identity=id,_hooked=lambda m:False,fallback_reason=lambda *a:None,get_attention_context=lambda name:(self.meta,None,None,None))
  exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),self.ns)
 def run_case(self):return self.ns['allowed'](self.layer,T())
 def test_decode_only(self):
  self.assertTrue(self.run_case())
  for k in vars(self.meta):
   old=getattr(self.meta,k);setattr(self.meta,k,1);self.assertFalse(self.run_case());setattr(self.meta,k,old)
 def test_aux_export(self):self.parent.aux_hidden_state_layers=(1,);self.assertFalse(self.run_case())
 def test_replacement(self):self.layer.post_attention_layernorm=self.NS(variance_size_override=None,has_weight=True,hidden_size=2048);self.assertFalse(self.run_case())
 def test_hooks(self):self.ns['_hooked']=lambda m:True;self.assertFalse(self.run_case())
 def test_variance_override(self):self.layer.post_attention_layernorm.variance_size_override=1024;self.assertFalse(self.run_case())
 def test_weightless(self):self.layer.post_attention_layernorm.has_weight=False;self.assertFalse(self.run_case())
 def test_mlp_reject(self):self.ns['fallback_reason']=lambda *a:'hooks';self.assertFalse(self.run_case())

if __name__=='__main__':unittest.main(verbosity=2)
