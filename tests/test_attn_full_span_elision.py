"""CPU control-flow tests; native GPU arithmetic/model proof belongs to the worker.

Load only the actual forward AST, without importing the plugin or initializing
an accelerator. The independent reference is the exact frozen base git object.
"""
import ast
import copy
from pathlib import Path
import subprocess
from types import SimpleNamespace as NS
import unittest

import torch
from torch.utils._python_dispatch import TorchDispatchMode

BASE = '5f39700683751f382e099bd638df4e3dddbc84b3'
FILE = 'vllm_fl/dispatch/backends/vendor/metax/impl/attention/flash_attn.py'
ROOT = Path(__file__).resolve().parents[1]
BASE_SOURCE = subprocess.check_output(
    ['git', '-C', str(ROOT), 'show', f'{BASE}:{FILE}'], text=True)


def forward_node(source):
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == 'FlashAttentionImpl')
    return copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                              and n.name == 'forward'))


class SliceCounter(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.count = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func == torch.ops.aten.slice.Tensor:
            self.count += 1
        return func(*args, **(kwargs or {}))


class SpecialTensor(torch.Tensor):
    pass


class Harness:
    """Deterministic native-call stand-ins; compare arguments, writes and aliasing.

    Stand-ins do not establish GPU kernel safety or numerical correctness.
    """
    def __init__(self, source, direct=True):
        self.direct = direct
        self.calls = []
        def snapshot(t):
            return (t.detach().clone(), tuple(t.stride()), t.storage_offset())
        def native(fn, q, k, v, out, *args):
            self.calls.append(('decode', snapshot(q), snapshot(out), args))
            if self.direct:
                out.copy_(q + k.sum() + v.sum())
            return self.direct
        def fallback(**kw):
            self.calls.append(('fallback', snapshot(kw['q'])))
            return kw['q'] + kw['k_cache'].sum() + kw['v_cache'].sum()
        def prefill(fn, q, k, v, out, *args):
            self.calls.append(('prefill', snapshot(q), snapshot(out)))
            out.copy_(q - 33)
            return True
        def cascade(out, q, *args, **kw):
            self.calls.append(('cascade', snapshot(q)))
            out.copy_(q - 44)
        self.types = NS(DECODER='decoder', ENCODER='encoder', ENCODER_ONLY='encoder-only')
        env = dict(torch=torch, FlashAttentionMetadata=object, AttentionType=self.types,
                   try_decode_into=native, flash_attn_with_kvcache=fallback,
                   reshape_query_for_spec_decode=lambda q, n: q,
                   reshape_attn_output_for_spec_decode=lambda q: q,
                   try_prefill_into=prefill, flash_attn_varlen_func=object(),
                   cascade_attention=cascade)
        node = forward_node(source)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                     FILE, 'exec'), env)
        self.forward = env['forward']
        self.impl = NS(attn_type='decoder', kv_sharing_target_layer_name=None,
                       kv_cache_dtype='auto', dcp_world_size=1, num_kv_heads=1,
                       scale=.5, sliding_window=(-1, -1), alibi_slopes=None,
                       logits_soft_cap=0., sinks=None, vllm_flash_attn_version=2)
        def other(q, *args, **kw):
            out = args[2] if self.impl.attn_type.startswith('encoder') else args[4]
            out.copy_(q - 55)
            self.calls.append(('other', snapshot(q)))
            return out
        self.impl._forward_encoder_attention = other
        self.impl._forward_with_dcp = other

    def run(self, q, out, count, prefills=0, actual=None, cascade=False):
        md = NS(num_actual_tokens=count if actual is None else actual,
                num_decode_tokens=count, num_decodes=max(1, count), num_prefills=prefills,
                use_cascade=cascade, query_start_loc=torch.tensor([0, count]),
                seq_lens=None, max_query_len=1, max_seq_len=8, block_table=None,
                scheduler_metadata=None, prefill_cu_seq_lens_k=object(),
                prefill_block_table=None, prefill_query_start_loc=None,
                prefill_max_seq_len=8, causal=True, decode_block_table=None,
                decode_seq_lens=None, cu_prefix_query_lens=None,
                prefix_kv_lens=None, suffix_kv_lens=None, common_prefix_len=0,
                max_num_splits=0, prefix_scheduler_metadata=None)
        scale = torch.ones(1)
        layer = NS(_q_scale=scale, _k_scale=scale, _v_scale=scale)
        cache = torch.arange(16., device='cpu').reshape(2, 2, 4)
        before = cache.clone()
        result = self.forward(self.impl, layer, q, q, q, cache, md, out,
                              cache_already_written=True)
        torch.testing.assert_close(cache, before, rtol=0, atol=0)
        return result


class FullSpanTests(unittest.TestCase):
    def compare(self, count=4, rows=4, prefills=0, actual=None, direct=True,
                strided=False, subclass=False, output_subclass=False,
                kind='decoder', dcp=1, cascade=False):
        q = torch.arange(max(rows, 1) * 32., device='cpu').reshape(max(rows, 1), 4, 8)
        q = q[:rows, :2, ::2] if strided else q[:rows, :2, :4].contiguous()
        if subclass:
            q = q.as_subclass(SpecialTensor)
        outputs, traces, counts = [], [], []
        for source in (BASE_SOURCE, (ROOT / FILE).read_text()):
            h = Harness(source, direct)
            h.impl.attn_type, h.impl.dcp_world_size = kind, dcp
            # Nonzero storage offset and noncontiguous destination are intentional.
            store = torch.full((rows + 2, 2, 8), -999., device='cpu')
            out = store[1:rows+1, :, ::2]
            if output_subclass:
                out = out.as_subclass(SpecialTensor)
            with SliceCounter() as counter:
                result = h.run(q, out, count, prefills, actual, cascade)
            if kind.startswith('encoder'):
                self.assertEqual(result.data_ptr(), out.data_ptr())
            else:
                self.assertIs(result, out)
            self.assertTrue(torch.all(store[0] == -999))
            self.assertTrue(torch.all(store[-1] == -999))
            self.assertTrue(torch.all(store[:, :, 1::2] == -999))
            outputs.append(store)
            traces.append(h.calls)
            counts.append(counter.count)
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        self.assertEqual(len(traces[0]), len(traces[1]))
        for a, b in zip(traces[0], traces[1]):
            self.assertEqual(a[0], b[0])
            for x, y in zip(a[1:], b[1:]):
                if x and isinstance(x[0], torch.Tensor):
                    torch.testing.assert_close(x[0], y[0], rtol=0, atol=0)
                    self.assertEqual(x[1:], y[1:])
                else:
                    self.assertEqual(x, y)
        return counts

    def test_full_decode_eliminates_exactly_two_slices(self):
        for m in (1, 2, 3, 4):
            for strided in (False, True):
                with self.subTest(m=m, strided=strided):
                    a, b = self.compare(count=m, rows=m, strided=strided)
                    self.assertEqual(a - b, 2)

    def test_partial_padded_zero_and_mixed_fallback(self):
        for kw in (dict(count=2), dict(count=0), dict(count=0, rows=0),
                   dict(count=2, prefills=1, actual=4),
                   dict(count=4, prefills=1, actual=4)):
            with self.subTest(kw=kw):
                a, b = self.compare(**kw)
                self.assertEqual(a, b)

    def test_subclass_encoder_dcp_cascade_unchanged(self):
        for kw in (dict(subclass=True), dict(output_subclass=True), dict(kind='encoder'),
                   dict(dcp=2), dict(cascade=True)):
            with self.subTest(kw=kw):
                a, b = self.compare(**kw)
                self.assertEqual(a, b)

    def test_native_reject_retains_original_fallback(self):
        a, b = self.compare(direct=False, strided=True)
        self.assertEqual(a - b, 2)

    def test_replacement_and_count_invalidation_on_same_impl(self):
        h = Harness((ROOT / FILE).read_text())
        for count, rows, bias in ((4, 4, 0), (2, 4, 100), (1, 1, 200), (0, 3, 300)):
            q = torch.full((rows, 2, 4), float(bias), device='cpu')
            out = torch.full_like(q, -999)
            h.run(q, out, count)
            expected = torch.full_like(q, -999)
            expected[:count] = q[:count] + torch.arange(16.).sum()
            torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_only_declared_host_expressions_changed(self):
        base = forward_node(BASE_SOURCE)
        node = forward_node((ROOT / FILE).read_text())
        class Undo(ast.NodeTransformer):
            def visit_Assign(self, n):
                if any(isinstance(t, ast.Name) and t.id == 'full_decode_span'
                       for t in n.targets):
                    return None
                return self.generic_visit(n)
            def visit_IfExp(self, n):
                if isinstance(n.test, ast.Name) and n.test.id == 'full_decode_span':
                    return n.orelse
                return self.generic_visit(n)
        self.assertEqual(ast.dump(base), ast.dump(Undo().visit(node)))


if __name__ == '__main__':
    unittest.main()
