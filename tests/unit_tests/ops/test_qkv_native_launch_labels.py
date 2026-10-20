"""Exercise actual caller handoff without importing torch/vLLM."""
import ast
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

SOURCE = Path(__file__).parents[3] / 'vllm_fl/models/minicpm5_rope_kv.py'


class LaunchLabels(unittest.TestCase):
    def test_actual_fused_caller_binds_layer_and_decode_only(self):
        tree = ast.parse(SOURCE.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'fused_qkv_attention')
        fn.decorator_list = []
        tensor = Mock(); tensor.view.return_value = tensor
        qkv = Mock(); qkv.split.return_value = [tensor, tensor, tensor]
        op = ModuleType('flag_gems.runtime.backend._metax.fused.qkv_rope_cache')
        op.try_qkv_rope_cache = Mock(return_value=qkv)
        layer = NS(impl=NS(forward=Mock()))
        cache = Mock(); cache.unbind.return_value = (object(), object())
        metadata = NS(slot_mapping=object())
        ns = dict(get_attention_context=lambda _: (metadata, layer, cache, None),
                  _qkv_observe=False, trace=Mock())
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(SOURCE), 'exec'), ns)
        with patch.dict(sys.modules, {op.__name__: op}):
            for i in range(42):
                attention = NS(qkv_proj=NS(weight=object()), rotary_emb=NS(cos_sin_cache=object()),
                               _qkv_launch_layer=i)
                self.assertTrue(ns['fused_qkv_attention'](attention, object(), object(), object(),
                                                         f'model.layers.{i}.self_attn.attn'))
                self.assertEqual(op.try_qkv_rope_cache.call_args.kwargs, dict(phase='decode', layer=i))
            op.try_qkv_rope_cache.return_value = None
            before = layer.impl.forward.call_count
            with self.assertRaisesRegex(RuntimeError, 'no fallback'):
                ns['fused_qkv_attention'](attention, object(), object(), object(), 'model.layers.41.self_attn.attn')
            self.assertEqual(layer.impl.forward.call_count, before)


if __name__ == '__main__':
    unittest.main()
