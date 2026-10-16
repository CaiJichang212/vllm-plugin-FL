"""CPU contract checks; deliberately import no GPU libraries."""
import importlib.util
from pathlib import Path
import sys
import threading
import types
import unittest
from unittest.mock import patch


class Tensor:
    dtype = 'bf16'
    requires_grad = False
    def __init__(self, rows=4, pointer=16):
        self.shape = (rows, 16, 128)
        self.pointer = pointer
        self.device = types.SimpleNamespace(index=0)
    def stride(self, dim=None):
        value = (2048, 128, 1)
        return value if dim is None else value[dim]
    def numel(self): return self.shape[0]*2048
    def data_ptr(self): return self.pointer


class Kernel:
    def __init__(self, owner): self.owner = owner
    def __getitem__(self, grid):
        def runner(*tensors):
            self.owner.calls.append(('hot', tensors, self.owner.stream, self.owner.hook))
        return runner


class JITFunction:
    debug = False
    launch_metadata = None
    cache_hook = None
    src = 'fixed source'
    cache_key = 'source hash'
    fn = staticmethod(lambda:None)
    compile = staticmethod(lambda:None)
    def __init__(self):
        self.pre_run_hooks = []; self.cache = {0:{}}; self.calls = []
        self.used_global_vals = {}; self.__globals__ = {'tl':object()}
        self.stream = 0; self.hook = None
        self.binder = lambda *args, **kw: ({}, ['ptr'], [args[-1]], list(args[:9]), kw)
    def run(self): pass
    def __getitem__(self, grid):
        def cold(*args, **kw):
            self.calls.append(('cold', args[:9], self.stream, self.hook))
            if any(values.get(name) != old for (name, _), (old,values) in self.used_global_vals.items()):
                raise RuntimeError('global changed')
            kw['debug']=self.debug
            _,sig,const,_,extra=self.binder(*args,**kw)
            key=''.join(sig)+str((const,extra))
            kernel=self.cache[0].setdefault(key,Kernel(self))
            return kernel
        return cold


class Launch(unittest.TestCase):
    def setUp(self):
        fake_triton=types.SimpleNamespace(__version__='3.0.0',runtime=types.SimpleNamespace(jit=types.SimpleNamespace(JITFunction=JITFunction)),knobs=types.SimpleNamespace(runtime=types.SimpleNamespace(jit_cache_hook=None,jit_post_compile_hook=None)))
        path=Path(__file__).resolve().parents[3]/'vllm_fl/dispatch/backends/vendor/metax/impl/attention/ops/rope_kv_launch.py'
        spec=importlib.util.spec_from_file_location('cpu_rope_launch',path)
        self.m=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,{'torch':types.SimpleNamespace(Tensor=Tensor),'vllm.triton_utils':types.SimpleNamespace(triton=fake_triton)}):spec.loader.exec_module(self.m)
        self.t=tuple(Tensor() for _ in range(9));self.e=JITFunction()
        self.m.diagnostics(True)
    def call(self): return self.m.launch(self.e,self.t,4,8)
    def test_cold_once_hot_current_arguments_stream_hooks(self):
        self.call();self.assertEqual(len(self.e.calls),1)
        self.t=tuple(Tensor(pointer=32) for _ in range(9));self.e.stream=7;self.e.hook=object()
        self.call();self.assertEqual([x[0] for x in self.e.calls],['cold','hot'])
        self.assertEqual(self.e.calls[-1][1],self.t);self.assertEqual(self.e.calls[-1][2],7)
        self.assertIs(self.e.calls[-1][3],self.e.hook)
        record=next(iter(self.m._cache(0).values()))
        self.assertFalse(any(isinstance(x,Tensor) for x in record))
    def test_shapes_strides_alignment_warps_are_distinct(self):
        self.call();self.t=tuple(Tensor(rows=5) for _ in range(9));self.call()
        self.t=tuple(Tensor(rows=5,pointer=17) for _ in range(9));self.call()
        self.m.launch(self.e,self.t,4,4)
        self.assertEqual([x[0] for x in self.e.calls],['cold']*4)
    def test_cache_replacement_and_entry_removal(self):
        self.call();self.e.cache[0]={};self.call()
        self.e.cache[0].clear();self.call()
        self.assertEqual([x[0] for x in self.e.calls],['cold']*3)
    def test_dynamic_hook_debug_and_grad_fallback(self):
        self.call();self.e.pre_run_hooks=[object()];self.call();self.e.pre_run_hooks=[]
        self.e.debug=True;self.call();self.e.debug=False
        self.t[0].requires_grad=True;self.call()
        self.assertEqual([x[0] for x in self.e.calls],['cold']*4)
    def test_global_change_preserves_original_failure(self):
        env={'x':1};self.e.used_global_vals={('x',id(env)):(1,env)}
        self.call();env['x']=2
        with self.assertRaisesRegex(RuntimeError,'global changed'):self.call()
        self.assertEqual(self.e.calls[-1][0],'cold')
    def test_jit_identity_source_binder_dependency_invalidation(self):
        self.call();self.e.src='changed';self.call()
        old=self.e.binder;self.e.binder=lambda *a,**k:old(*a,**k);self.call()
        self.e.__globals__['tl']=object();self.call()
        self.assertEqual([x[0] for x in self.e.calls],['cold']*4)
    def test_lru_bounded_and_pid_clear(self):
        for i in range(65):
            self.t=tuple(Tensor(rows=i+1) for _ in range(9));self.call()
        self.assertEqual(len(self.m._cache(0)),64)
        self.assertEqual(self.m.diagnostics()['evicted'],1)
        with patch.object(self.m.os,'getpid',return_value=-1):self.assertEqual(len(self.m._cache(0)),0)
    def test_thread_and_device_isolation(self):
        self.call();values=[]
        thread=threading.Thread(target=lambda:values.append(len(self.m._cache(0))))
        thread.start();thread.join();self.assertEqual(values,[0])
        self.assertEqual(len(self.m._cache(1)),0)
    def test_unsupported_compile_state_and_tensor_subclass(self):
        self.m.triton.knobs.runtime.jit_cache_hook=object();self.call()
        self.m.triton.knobs.runtime.jit_cache_hook=None
        class Sub(Tensor):pass
        self.t=(Sub(),*self.t[1:]);self.call()
        self.assertEqual([x[0] for x in self.e.calls],['cold','cold'])


if __name__=='__main__': unittest.main()
