"""CPU ABI/support checks. These mocks do not validate native GPU semantics."""
import importlib.util
from pathlib import Path
import types
import sys
import unittest
from unittest.mock import patch

class Tensor:
    serial = 0
    def __init__(self, shape, dtype='bf16', strides=None, ptr=None):
        self.shape=tuple(shape);self.ndim=len(shape);self.dtype=dtype
        self.device=types.SimpleNamespace(type='cuda');self.requires_grad=False
        Tensor.serial+=1;self.ptr=Tensor.serial*100000000 if ptr is None else ptr
        size=1;values=[]
        for n in reversed(shape):values.append(size);size*=n
        self.strides=tuple(reversed(values)) if strides is None else tuple(strides)
    def stride(self, i=None):return self.strides if i is None else self.strides[i]
    def data_ptr(self):return self.ptr
    def element_size(self):return 2 if self.dtype=='bf16' else 4
    def is_contiguous(self):
        size=1
        for n,s in zip(reversed(self.shape),reversed(self.strides)):
            if n>1 and s!=size:return False
            size*=n
        return True

class Tests(unittest.TestCase):
    def setUp(self):
        self.calls=[];self.grad=False;self.failure=False;self.wrong=False
        def public(*a,**k):raise AssertionError('caller owns fallback')
        def native(*args):
            self.calls.append(args)
            if self.failure:raise RuntimeError('native failed')
            return (Tensor((1020,16,128)) if self.wrong else args[3],)*9
        self.fa=types.SimpleNamespace(flash_attn_varlen_func=public,FlashAttnVarlenFunc=types.SimpleNamespace(forward=object()),_flash_attn_varlen_forward=object(),maybe_contiguous=lambda t:t,flash_attn_cuda=types.SimpleNamespace(varlen_fwd=native))
        torch=types.SimpleNamespace(Tensor=Tensor,bfloat16='bf16',int32='i32',is_grad_enabled=lambda:self.grad)
        flash=types.SimpleNamespace(__version__='2.6.3',flash_attn_interface=self.fa)
        source=Path(__file__).resolve().parents[3]/'vllm_fl/dispatch/backends/vendor/metax/impl/attention/ops/prefill_output.py'
        with patch.dict(sys.modules,{'torch':torch,'flash_attn':flash}):
            spec=importlib.util.spec_from_file_location('prefill_test',source);self.m=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.m)
        self.args=[public,Tensor((1020,16,128),strides=(2560,128,1)),Tensor((512,16,2,128)),Tensor((512,16,2,128)),Tensor((1020,16,128)),Tensor((4,256),'i32'),Tensor((5,),'i32'),Tensor((5,),'i32'),1024,4096,128**-.5,True,(-1,-1),None,0.,None]
    def test_native_argument_mapping_and_offset_destination(self):
        self.args[4].ptr+=128
        self.assertTrue(self.m.try_prefill_into(*self.args));c=self.calls[0]
        self.assertEqual(c[:6],tuple(self.args[1:5]+self.args[6:8]))
        self.assertEqual(c[6:10],(None,None,self.args[5],None))
        self.assertEqual(c[10:],(1024,4096,0.,128**-.5,False,True,-1,-1,0.,False,None,None,False))
    def test_actual_small_prefill_tail_and_single_query_fallback(self):
        for n in (2,4,15,16,17):
            args=list(self.args);args[1]=Tensor((n,16,128));args[4]=Tensor((n,16,128));args[8]=n
            self.assertTrue(self.m.try_prefill_into(*args))
        args=list(self.args);args[8]=1
        before=len(self.calls);self.assertFalse(self.m.try_prefill_into(*args));self.assertEqual(len(self.calls),before)
    def test_rejections_do_not_submit(self):
        values=[(0,lambda:None),(1,Tensor((0,16,128))),(1,Tensor((1020,16,128),'fp16')),(4,Tensor((1020,16,128),strides=(4096,128,1))),(5,Tensor((4,256),'i64')),(6,Tensor((4,),'i32')),(8,0),(9,16385),(12,(16,0)),(13,object()),(14,1.),(15,object())]
        for index,value in values:
            args=list(self.args);args[index]=value
            self.assertFalse(self.m.try_prefill_into(*args),str(index))
        self.grad=True;self.assertFalse(self.m.try_prefill_into(*self.args));self.grad=False
        self.args[1].requires_grad=True;self.assertFalse(self.m.try_prefill_into(*self.args))
        self.assertFalse(self.calls)
    def test_all_input_aliases_rejected(self):
        for i in (1,2,3,5,6,7):
            args=list(self.args);args[4]=Tensor(args[1].shape,ptr=args[i].ptr)
            self.assertFalse(self.m.try_prefill_into(*args))
        self.assertFalse(self.calls)
    def test_subclass_hooks_and_collection(self):
        class Sub(Tensor):pass
        args=list(self.args);args[1]=Sub((1020,16,128))
        self.assertFalse(self.m.try_prefill_into(*args))
        with patch.dict('os.environ',{'MHA_INPUT_COLLECTION':'1'}):self.assertFalse(self.m.try_prefill_into(*self.args))
        self.fa._flash_attn_varlen_forward=object();self.assertFalse(self.m.try_prefill_into(*self.args))
        self.assertFalse(self.calls)
    def test_native_error_and_wrong_destination_do_not_retry(self):
        self.failure=True
        with self.assertRaisesRegex(RuntimeError,'native failed'):self.m.try_prefill_into(*self.args)
        self.assertEqual(len(self.calls),1)
        self.failure=False;self.wrong=True
        with self.assertRaisesRegex(RuntimeError,'destination'):self.m.try_prefill_into(*self.args)

if __name__=='__main__':unittest.main()
