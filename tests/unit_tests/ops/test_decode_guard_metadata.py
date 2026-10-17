"""Differential CPU support/alias/ABI checks against the pinned parent adapter.

These fake Tensor contracts do not establish native GPU correctness.
"""
import copy
from collections import Counter
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

BASE = '3e47a37b4534e47ca848325ea78cdd88fd3c5556'
REL = 'vllm_fl/dispatch/backends/vendor/metax/impl/attention/ops/decode_output.py'


class Tensor:
    def __init__(self, shape, dtype='bf16', ptr=100000, stride=None):
        self.shape = tuple(shape)
        self.ndim = len(shape)
        self.dtype = dtype
        self.device = types.SimpleNamespace(type='cuda')
        self.requires_grad = False
        self.ptr = ptr
        strides = []; size = 1
        for n in reversed(shape):
            strides.append(size); size *= n
        self.strides = tuple(reversed(strides)) if stride is None else tuple(stride)
        self.reads = Counter()
    def __getattribute__(self, name):
        if name in ('shape', 'device') and 'reads' in object.__getattribute__(self, '__dict__'):
            object.__getattribute__(self, 'reads')[name] += 1
        return object.__getattribute__(self, name)
    def stride(self, i=None):
        self.reads['stride'] += 1
        return self.strides if i is None else self.strides[i]
    def data_ptr(self):
        self.reads['pointer'] += 1
        return self.ptr
    def element_size(self): return 2 if self.dtype == 'bf16' else 4
    def is_contiguous(self):
        size = 1
        for n, s in zip(reversed(self.shape), reversed(self.strides)):
            if n > 1 and s != size: return False
            size *= n
        return True
    def as_strided(self, shape, stride): return Tensor(shape, self.dtype, self.ptr, stride)


class GuardTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[3]
        parent = subprocess.check_output(['git', '-C', str(root), 'show', BASE+':'+REL], text=True)
        self.modules = []
        self.calls = []
        def public(*a, **kw): raise AssertionError('fallback must be returned to caller')
        self.public = public
        self.fa = types.SimpleNamespace(flash_attn_with_kvcache=public, maybe_contiguous=lambda t:t)
        def native(*args):
            self.calls.append(args)
            return args[12], None
        self.fa.flash_attn_cuda = types.SimpleNamespace(fwd_kvcache=native)
        flash = types.SimpleNamespace(__version__='2.6.3', flash_attn_interface=self.fa)
        torch = types.SimpleNamespace(Tensor=Tensor, bfloat16='bf16', int32='int32')
        with patch.dict(sys.modules, {'torch':torch, 'flash_attn':flash}):
            for i, source in enumerate((parent, (root/REL).read_text())):
                m = types.ModuleType('adapter'+str(i)); exec(compile(source, REL, 'exec'), m.__dict__)
                self.modules.append(m)
    def args(self):
        return [self.public, Tensor((4,1,16,128), ptr=100000, stride=(2560,2048,128,1)),
                Tensor((8,16,2,128), ptr=1000000), Tensor((8,16,2,128), ptr=2000000),
                Tensor((4,16,128), ptr=3000000), Tensor((4,32), 'int32', ptr=4000000),
                Tensor((4,), 'int32', ptr=5000000), 128**-.5, (-1,-1), None, 0., None]
    def compare(self, args, expected=None):
        rows=[]
        for m in self.modules:
            local=[copy.deepcopy(x) if isinstance(x,Tensor) else x for x in args]
            self.calls.clear(); m.diagnostics(True)
            hit=m.try_decode_into(*local)
            rows.append((hit, [(c[12].shape,c[12].strides,c[12].ptr,c[13:]) for c in self.calls]))
            m.diagnostics(False)
            if expected is not None: self.assertEqual(hit,expected)
        self.assertEqual(rows[0],rows[1])
    def test_supported_batches_offsets_and_metadata_reads(self):
        for batch in range(1,9):
            for offset in (0,74):
                a=self.args(); a[1]=Tensor((batch,1,16,128),ptr=100000+offset,stride=(2560,2048,128,1))
                a[4]=Tensor((batch,16,128),ptr=3000000+offset)
                a[5]=Tensor((batch,32),'int32',ptr=4000000+offset)
                a[6]=Tensor((batch,),'int32',ptr=5000000+offset)
                self.compare(a,True)
        counts=[]
        for m in self.modules:
            a=self.args();self.assertTrue(m.try_decode_into(*a))
            counts.append(Counter({k:sum(t.reads[k] for t in a[1:7]) for k in ('shape','stride','device','pointer')}))
        for k in counts[0]:self.assertLess(counts[1][k],counts[0][k],k)
    def test_rejections_and_tensor_subclass(self):
        class SubTensor(Tensor): pass
        values=[(1,SubTensor((4,1,16,128))), (1,Tensor((0,1,16,128))), (1,Tensor((9,1,16,128))),
                (1,Tensor((4,2,16,128))), (1,Tensor((4,1,16,128),'fp16')),
                (4,Tensor((4,16,128),stride=(4096,128,1))), (5,Tensor((4,32),'int64')),
                (5,Tensor((4,32),'int32',stride=(0,1))), (6,Tensor((4,),'int32',stride=(0,))),
                (8,(16,0)), (9,object()), (10,1.), (11,object())]
        for index,value in values:
            a=self.args();a[index]=value;self.compare(a,False)
        for index in range(1,7):
            a=self.args();a[index].device=types.SimpleNamespace(type='cpu');self.compare(a,False)
        for index in range(1,5):
            a=self.args();a[index].requires_grad=True;self.compare(a,False)
        for index in (1,5,6):
            a=self.args();a[index].strides=tuple(-s for s in a[index].strides);self.compare(a,False)
    def test_adjacent_overlapping_intervals_and_empty_table(self):
        for index in range(1,7):
            if index==4:continue
            for delta in (-2,0,2):
                a=self.args();begin,end=self.modules[0]._interval(a[index])
                a[4].ptr=end+delta;self.compare(a,delta>=0)
                a=self.args();begin,end=self.modules[0]._interval(a[index])
                a[4].ptr=begin-4*16*128*2+delta;self.compare(a,delta<=0)
        # Preserve even unusual original-domain shapes; do not invent a guard.
        a=self.args();a[5]=Tensor((4,0),'int32',ptr=4000000);self.compare(a)
    def test_hooks_collection_and_diagnostics(self):
        for attr in ('flash_attn_with_kvcache','maybe_contiguous'):
            with patch.object(self.fa,attr,lambda *a:None):self.compare(self.args(),False)
        with patch.dict(os.environ,{'MHA_INPUT_COLLECTION':'1'}):self.compare(self.args(),False)
        for m in self.modules:
            m.diagnostics(False);before=m.diagnostics(False)
            self.assertTrue(m.try_decode_into(*self.args()));self.assertEqual(m.diagnostics(False),before)
    def test_native_storage_and_failure_are_not_retried(self):
        for m in self.modules:
            with patch.object(self.fa.flash_attn_cuda,'fwd_kvcache',side_effect=RuntimeError('native error')) as fn:
                with self.assertRaisesRegex(RuntimeError,'native error'):m.try_decode_into(*self.args())
                self.assertEqual(fn.call_count,1)
            with patch.object(self.fa.flash_attn_cuda,'fwd_kvcache',return_value=(Tensor((4,1,16,128),ptr=999),None)):
                with self.assertRaisesRegex(RuntimeError,'destination storage'):m.try_decode_into(*self.args())


if __name__=='__main__': unittest.main()
