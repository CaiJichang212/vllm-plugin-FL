"""CPU contracts for actual adapter/selector source; no GPU evidence or imports.

Run directly with python3 tests/cpu_contract/test_iluvatar_paged_prefill.py.
Do not use pytest here: repository-wide conftest imports the device runtime.
"""
import ast
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
ADAPTER = ROOT / 'vllm_fl/dispatch/backends/flaggems/impl/iluvatar_paged_prefill.py'
NS = types.SimpleNamespace


def module(name, **attrs):
    result = types.ModuleType(name)
    result.__dict__.update(attrs)
    return result


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class Tensor:
    def __init__(self, shape, dtype='bf16', strides=None, device=None):
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = dtype
        self.device = device or NS(type='cuda', index=0)
        contiguous, stride = [], 1
        for dim in reversed(shape):
            contiguous.insert(0, stride)
            stride *= dim
        self.strides = strides or tuple(contiguous)

    def stride(self):
        return self.strides

    def unbind(self, axis):
        assert axis == 1
        return tuple(Tensor(self.shape[:1] + self.shape[2:], self.dtype,
                            self.strides[:1] + self.strides[2:], self.device) for _ in range(2))

    def __bool__(self):
        raise AssertionError('device bool forbidden')

    def item(self):
        raise AssertionError('device scalar read forbidden')


class Original:
    def __init__(self, **kwargs):
        self.__dict__.update(dict(num_heads=16, num_kv_heads=2, head_size=128,
            scale=0.088, attn_type='decoder', kv_cache_dtype='auto',
            _is_per_token_head_quant=False, alibi_slopes=None, use_alibi_sqrt=False,
            sliding_window=(-1, -1), logits_soft_cap=0, sinks=None,
            chunk_lookback=-1, kv_sharing_target_layer_name=None, use_td=False))
        self.__dict__.update(kwargs)
        self.original_calls = []

    def forward(self, *args, **kwargs):
        self.original_calls.append((args, kwargs))
        return args[6]

    def do_kv_cache_update(self):
        raise AssertionError('forward must not update KV')


class Backend:
    forward_includes_kv_cache_update = False
    get_builder_cls = object()
    get_kv_cache_shape = object()
    get_kv_cache_stride_order = object()


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.kernel = Mock()
        self.platform = NS(vendor_name='iluvatar')
        self.config = NS(parallel_config=NS(decode_context_parallel_size=1,
            prefill_context_parallel_size=1), cache_config=NS(kv_sharing_fast_prefill=False))
        self.envs = NS(VLLM_BATCH_INVARIANT=False)
        self.modules = {
            'torch': module('torch', bfloat16='bf16', int32='i32', int64='i64',
                            cuda=NS(is_available=lambda: True)),
            'vllm': module('vllm', envs=self.envs),
            'vllm.config': module('vllm.config', get_current_vllm_config_or_none=lambda: self.config),
            'vllm.platforms': module('vllm.platforms', current_platform=self.platform),
            'vllm.v1.attention.backend': module('backend', AttentionType=NS(DECODER='decoder')),
            'vllm.v1.attention.backends.triton_attn': module('triton_attn',
                TritonAttentionBackend=Backend, TritonAttentionImpl=Original),
            'flag_gems.runtime.backend._iluvatar.fused.paged_prefill_attention':
                module('kernel', paged_prefill_attention=self.kernel),
        }
        self.patcher = patch.dict(sys.modules, self.modules)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.adapter = load(ADAPTER, 'candidate_test')
        self.impl = self.adapter.IluvatarPagedPrefillImpl()
        self.q, self.out = Tensor((24,16,128)), Tensor((24,16,128))
        self.cache = Tensor((20,2,16,2,128))
        self.meta = NS(causal=True, use_cascade=False, max_query_len=17,
            num_actual_tokens=20, max_seq_len=65, query_start_loc=Tensor((4,), 'i32'),
            seq_lens=Tensor((3,), 'i32'), block_table=Tensor((3,5), 'i32'))

    def call(self, **kwargs):
        args = dict(layer=object(), query=self.q, key=object(), value=object(),
                    kv_cache=self.cache, attn_metadata=self.meta, output=self.out)
        args.update(kwargs)
        return self.impl.forward(**args)

    def test_actual_module_inherits_layout_builder_and_update(self):
        cls = self.adapter.IluvatarPagedPrefillBackend
        for name in ('get_builder_cls', 'get_kv_cache_shape', 'get_kv_cache_stride_order'):
            self.assertIs(getattr(cls, name), getattr(Backend, name))
        self.assertFalse(cls.forward_includes_kv_cache_update)
        self.assertIs(self.impl.do_kv_cache_update.__func__, Original.do_kv_cache_update)
        self.assertIs(cls.get_impl_cls(), type(self.impl))

    def test_mixed_prefill_padded_output_and_exact_api(self):
        self.assertIs(self.call(), self.out)
        self.assertFalse(self.impl.original_calls)
        args, kwargs = self.kernel.call_args
        self.assertIs(args[0], self.q)
        self.assertIs(args[3], self.out)
        self.assertEqual(args[1].shape, (20,16,2,128))
        self.assertEqual(args[4:], (self.meta.query_start_loc,self.meta.seq_lens,
                                  self.meta.block_table,17,0.088))
        self.assertEqual(kwargs, {'num_actual_tokens':20})

    def test_positive_noncontiguous_strides_and_int64(self):
        self.q.strides = (8192,256,2)
        self.out.strides = (8192,256,2)
        self.cache.strides = (8192,4096,128,2048,1)
        self.meta.block_table.strides = (20,2)
        self.meta.block_table.dtype = 'i64'
        self.call()
        self.kernel.assert_called_once()

    def test_metadata_guards_fallback_without_device_read(self):
        cases = {'causal':[False,Tensor((1,))], 'use_cascade':[True,Tensor((1,))],
                 'max_query_len':[1,0,None], 'num_actual_tokens':[0,25,None],
                 'max_seq_len':[0,81,None], 'mm_prefix_range':[[]],
                 'mm_prefix_range_tensor':[Tensor((1,))]}
        for name, values in cases.items():
            original = getattr(self.meta,name,None)
            for value in values:
                with self.subTest(name=name,value=type(value)):
                    setattr(self.meta,name,value)
                    self.assertIs(self.call(),self.out)
            setattr(self.meta,name,original)
        self.kernel.assert_not_called()

    def test_none_metadata_and_quant_output_fallback(self):
        self.call(attn_metadata=None)
        self.call(output_scale=object())
        self.call(output_block_scale=object())
        self.kernel.assert_not_called()
        self.assertEqual(len(self.impl.original_calls),3)

    def test_layout_dtype_device_stride_guard(self):
        for argument, tensor in [('query',Tensor((24,8,128))),
                ('query',Tensor((24,16,128),'fp16')),
                ('output',Tensor((19,16,128))),
                ('output',Tensor((25,16,128))),
                ('output',Tensor((24,16,128),strides=(0,128,1))),
                ('kv_cache',Tensor((20,2,8,2,128))),
                ('query',Tensor((24,16,128),device=NS(type='cpu',index=0)))]:
            with self.subTest(argument=argument,shape=tensor.shape):
                self.call(**{argument:tensor})
        self.kernel.assert_not_called()

    def test_static_special_semantics_disable(self):
        for name, value in dict(num_heads=8, num_kv_heads=4, head_size=64,
                kv_cache_dtype='fp8', _is_per_token_head_quant=True, alibi_slopes=[1],
                use_alibi_sqrt=True, sliding_window=(127,0), logits_soft_cap=10,
                sinks=object(),chunk_lookback=0,kv_sharing_target_layer_name=0,
                use_td=True,attn_type='encoder',scale=float('nan')).items():
            with self.subTest(name=name):
                self.assertFalse(self.adapter.IluvatarPagedPrefillImpl(**{name:value})._paged_prefill_enabled)

    def test_parallel_platform_batch_invariance_disable(self):
        for obj, name, value in [(self.platform,'vendor_name','metax'),
                (self.envs,'VLLM_BATCH_INVARIANT',True),
                (self.config.parallel_config,'decode_context_parallel_size',2),
                (self.config.parallel_config,'prefill_context_parallel_size',2),
                (self.config.cache_config,'kv_sharing_fast_prefill',True)]:
            old = getattr(obj,name)
            setattr(obj,name,value)
            self.assertFalse(self.adapter.IluvatarPagedPrefillImpl()._paged_prefill_enabled)
            setattr(obj,name,old)

    def test_kernel_error_propagates(self):
        self.kernel.side_effect = RuntimeError('compile failed')
        with self.assertRaisesRegex(RuntimeError,'compile failed'):
            self.call()
        self.assertFalse(self.impl.original_calls)

    def test_real_selector_preserves_legacy_vendor_and_errors(self):
        self.modules.update({
            'vllm_fl.dispatch.backends.base':module('base',Backend=object),
            'vllm.v1.attention.backends.registry':module('registry',
                AttentionBackendEnum=NS(TRITON_ATTN=NS(get_path=lambda:'original.triton')))})
        with patch.dict(sys.modules,self.modules), patch.dict(os.environ,{},clear=True):
            selector = load(ROOT/'vllm_fl/dispatch/backends/flaggems/flaggems.py','selector_test').FlagGemsBackend()
            self.assertTrue(selector.attention_backend().endswith('IluvatarPagedPrefillBackend'))
            self.platform.vendor_name = 'metax'
            self.assertEqual(selector.attention_backend(),'original.triton')
            os.environ['VLLM_FL_USE_FLAGGEMS_ATTN']='1'
            self.assertTrue(selector.attention_backend().endswith('AttentionFLBackend'))
            with self.assertRaises(NotImplementedError): selector.attention_backend(use_mla=True)
            with self.assertRaises(ValueError): selector.attention_backend(use_sparse=True)
            self.modules['torch'].cuda.is_available=lambda:False
            with self.assertRaises(RuntimeError): selector.attention_backend()

    def test_actual_platform_method_explicit_backend_does_not_dispatch(self):
        tree=ast.parse((ROOT/'vllm_fl/platform.py').read_text())
        method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='get_attn_backend_cls')
        method.decorator_list=[]
        method.returns=None
        for arg in method.args.args: arg.annotation=None
        dispatch=Mock(return_value='policy.vendor')
        namespace={'logger':NS(info_once=Mock(), info=Mock())}
        exec(compile(ast.Module(body=[method],type_ignores=[]),'platform.py','exec'),namespace)
        selected=NS(get_class=lambda:NS(validate_configuration=lambda **kw:[]),get_path=lambda:'explicit.backend')
        config=NS(_asdict=lambda:{},use_mla=False,use_sparse=False)
        cls=NS(get_device_capability=lambda:None)
        with patch.dict(sys.modules,{'vllm_fl.dispatch':module('dispatch',call_op=dispatch)}):
            self.assertEqual(namespace['get_attn_backend_cls'](cls,selected,config),'explicit.backend')
            dispatch.assert_not_called()
            self.assertEqual(namespace['get_attn_backend_cls'](cls,None,config),'policy.vendor')
            dispatch.assert_called_once_with('attention_backend',use_mla=False,use_sparse=False)
            selected.get_class=lambda:NS(validate_configuration=lambda **kw:['unsupported'])
            with self.assertRaisesRegex(ValueError,'incompatible'):
                namespace['get_attn_backend_cls'](cls,selected,config)


if __name__ == '__main__':
    unittest.main()
