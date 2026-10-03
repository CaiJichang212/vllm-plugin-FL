# SPDX-License-Identifier: Apache-2.0
"""MetaX batch metadata, paged attention and graph lifecycle regressions."""

from types import SimpleNamespace

import pytest
import torch

# This backend imports MACA native operators and must not load on other vendors.
if "metax" not in torch.__version__:
    pytest.skip("Requires the competition MACA environment", allow_module_level=True)
pytest.importorskip("flash_attn")

from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm_fl.dispatch.backends.vendor.metax.impl.attention import flash_attn as fa


def builder(device="cpu"):
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            get_num_attention_heads=lambda _: 16,
            get_num_kv_heads=lambda _: 2,
            get_head_size=lambda: 128,
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1, cp_kv_cache_interleave_size=1,
        ),
        cache_config=SimpleNamespace(cache_dtype="auto"),
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(has_full_cudagraphs=lambda: False),
            max_cudagraph_capture_size=16,
        ),
        speculative_config=None,
    )
    spec = SimpleNamespace(dtype=torch.bfloat16, block_size=16)
    return fa.FlashAttentionMetadataBuilder(spec, ["layer0", "layer1"], config,
                                           torch.device(device))


def common(query_lens, seq_lens, device="cpu", cpu_copy=True):
    starts = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(0, dtype=torch.int32)
    seq_cpu = torch.tensor(seq_lens, dtype=torch.int32)
    return CommonAttentionMetadata(
        query_start_loc=starts.to(device), query_start_loc_cpu=starts,
        seq_lens=seq_cpu.to(device), _seq_lens_cpu=seq_cpu.clone() if cpu_copy else None,
        # Deliberately incorrect: upper bounds are not an exact-length source.
        seq_lens_cpu_upper_bound=seq_cpu + 100,
        num_reqs=len(query_lens), num_actual_tokens=sum(query_lens),
        max_query_len=max(query_lens), max_seq_len=max(seq_lens),
        block_table_tensor=torch.zeros((len(query_lens), 1), dtype=torch.int32, device=device),
        slot_mapping=torch.zeros(sum(query_lens), dtype=torch.int64, device=device),
    )


@pytest.mark.parametrize("query_lens,seq_lens,expected", [
    ([129], [129], [0, 129]),
    ([256, 385], [1024, 901], [0, 1024, 1925]),
    ([1, 129, 257], [511, 700, 1025], [0, 700, 1725]),
    ([128], [512], None),
    ([1, 1], [512, 700], None),
    ([1, 0], [512, 0], None),
    ([128, 129], [512, 513], [0, 513]),
])
def test_partition_and_exact_kv_lengths(query_lens, seq_lens, expected):
    metadata = builder().build(0, common(query_lens, seq_lens))
    actual = metadata.prefill_cu_seq_lens_k
    if expected is None:
        assert actual is None
    else:
        assert actual.dtype == torch.int32
        assert actual.is_contiguous()
        assert actual.tolist() == expected
        legacy = torch.tensor([0] + metadata.prefill_seq_lens.tolist(), dtype=torch.int32).cumsum(0, dtype=torch.int32)
        torch.testing.assert_close(actual, legacy, atol=0, rtol=0)


def test_lifetime_reorder_removal_and_microbatches():
    build = builder()
    cm = common([256, 385], [1024, 901])
    first = build.build(0, cm)
    saved = first.prefill_cu_seq_lens_k.clone()
    cm._seq_lens_cpu[:] = torch.tensor([901, 1024])
    cm.seq_lens[:] = cm._seq_lens_cpu
    second = build.build(0, cm)
    third = build.build(0, common([129], [777]))
    torch.testing.assert_close(first.prefill_cu_seq_lens_k, saved, atol=0, rtol=0)
    assert second.prefill_cu_seq_lens_k.tolist() == [0, 901, 1925]
    assert third.prefill_cu_seq_lens_k.tolist() == [0, 777]
    assert len({x.prefill_cu_seq_lens_k.data_ptr() for x in (first, second, third)}) == 3


def test_int32_boundary_and_overflow():
    limit = torch.iinfo(torch.int32).max
    assert builder().build(0, common([129], [limit])).prefill_cu_seq_lens_k.tolist() == [0, limit]
    with pytest.raises(ValueError, match="int32"):
        builder().build(0, common([129, 129], [limit, 129]))


def test_capture_and_fast_build():
    build = builder()
    assert build.build_for_cudagraph_capture(common([1, 0], [512, 0])).prefill_cu_seq_lens_k is None
    metadata = build.build_for_cudagraph_capture(common([256], [512]))
    assert metadata.prefill_cu_seq_lens_k.tolist() == [0, 512]
    assert build.build(0, common([129], [700]), fast_build=True).prefill_cu_seq_lens_k.tolist() == [0, 700]


@pytest.mark.gpu
def test_missing_cpu_copy():
    cm = common([1, 129], [100, 700], "cuda", cpu_copy=False)
    metadata = builder("cuda").build(0, cm)
    assert metadata.prefill_cu_seq_lens_k.device == cm.seq_lens.device
    assert metadata.prefill_cu_seq_lens_k.tolist() == [0, 700]


@pytest.mark.gpu
def test_cpu_accumulation_with_cuda_default_device():
    cm = common([129], [700], "cuda")
    build = builder("cuda")
    with torch.device("cuda"):
        metadata = build.build(0, cm)
    assert metadata.prefill_cu_seq_lens_k.tolist() == [0, 700]


def attention_inputs(query_lens, seq_lens):
    torch.manual_seed(20261001)
    cm = common(query_lens, seq_lens, "cuda")
    blocks_per_request = (max(seq_lens) + 15) // 16
    num_blocks = len(seq_lens) * blocks_per_request
    table = torch.randperm(num_blocks, device="cuda").to(torch.int32).reshape(len(seq_lens), -1)
    slots = []
    for row, (qlen, slen) in enumerate(zip(query_lens, seq_lens)):
        positions = torch.arange(slen - qlen, slen, device="cuda")
        slots.append(table[row, positions // 16].long() * 16 + positions % 16)
    cm.block_table_tensor = table
    cm.slot_mapping = torch.cat(slots)
    shape = (sum(query_lens),)
    q = torch.randn(*shape, 16, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(*shape, 2, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    cache = torch.randn(2, num_blocks, 16, 2, 128, device="cuda", dtype=torch.bfloat16)
    impl = fa.FlashAttentionImpl(16, 128, 128 ** -0.5, 2, None, None, "auto")
    scale = torch.ones((), device="cuda")
    layer = SimpleNamespace(_k_scale=scale, _v_scale=scale, _q_scale=scale)
    return cm, q, k, v, cache, impl, layer


@pytest.mark.gpu
@pytest.mark.parametrize("query_lens,seq_lens", [
    ([129], [129]), ([256, 385], [513, 901]),
    ([1, 129, 257], [511, 700, 1025]),
    ([128], [512]), ([1, 1], [512, 700]),
])
def test_paged_attention_outputs_and_kv(query_lens, seq_lens):
    cm, q, k, v, cache, impl, layer = attention_inputs(query_lens, seq_lens)
    metadata = builder("cuda").build(0, cm)
    initial = cache.clone()
    expected_cache = initial.clone()
    expected_cache[0].view(-1, 2, 128)[cm.slot_mapping] = k
    expected_cache[1].view(-1, 2, 128)[cm.slot_mapping] = v
    output = torch.empty_like(q)
    impl.forward(layer, q, k, v, cache, metadata, output=output)
    torch.testing.assert_close(cache, expected_cache, atol=0, rtol=0)

    # The pre-change varlen call uses the same cumulative KV lengths for all rows.
    legacy_cu = torch.tensor([0] + cm.seq_lens.tolist(), dtype=torch.int32,
                             device="cuda").cumsum(0, dtype=torch.int32)
    legacy = fa.flash_attn_varlen_func(
        q=q, k=cache[0], v=cache[1], block_table=cm.block_table_tensor,
        cu_seqlens_q=cm.query_start_loc, cu_seqlens_k=legacy_cu,
        max_seqlen_q=max(query_lens), max_seqlen_k=max(seq_lens),
        softmax_scale=128 ** -0.5, causal=True,
    )
    torch.testing.assert_close(output, legacy, atol=0.02, rtol=0.02)
    if metadata.num_decodes == 0:
        torch.testing.assert_close(output, legacy, atol=0, rtol=0)
    offset = 0
    for row, (qlen, slen) in enumerate(zip(query_lens, seq_lens)):
        keys = cache[0, cm.block_table_tensor[row].long()].reshape(-1, 2, 128)[:slen]
        values = cache[1, cm.block_table_tensor[row].long()].reshape(-1, 2, 128)[:slen]
        qh = q[offset:offset + qlen].transpose(0, 1).float()
        kh = keys.repeat_interleave(8, dim=1).transpose(0, 1).float()
        vh = values.repeat_interleave(8, dim=1).transpose(0, 1).float()
        scores = qh @ kh.transpose(-1, -2) * (128 ** -0.5)
        mask = torch.arange(slen, device="cuda")[None, :] > (
            torch.arange(qlen, device="cuda")[:, None] + slen - qlen)
        expected = (scores.masked_fill(mask, -torch.inf).softmax(-1) @ vh).transpose(0, 1).to(q.dtype)
        torch.testing.assert_close(output[offset:offset + qlen], expected, atol=0.02, rtol=0.02)
        offset += qlen
    # Same metadata, another layer's independent cache: no shared write state.
    next_cache = initial.clone()
    next_output = torch.empty_like(q)
    impl.forward(layer, q, k, v, next_cache, metadata, output=next_output)
    torch.testing.assert_close(next_cache, cache, atol=0, rtol=0)
    torch.testing.assert_close(next_output, output, atol=0, rtol=0)


@pytest.mark.gpu
def test_uniform_decode_graph_replay():
    cm, q, k, v, cache, impl, layer = attention_inputs([1, 1], [126, 254])
    metadata = builder("cuda").build_for_cudagraph_capture(cm)
    assert metadata.prefill_cu_seq_lens_k is None
    output = torch.empty_like(q)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            impl.forward(layer, q, k, v, cache, metadata, output=output)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        impl.forward(layer, q, k, v, cache, metadata, output=output)
    for _ in range(2):
        q.normal_()
        k.normal_()
        cm.seq_lens.add_(1)
        positions = cm.seq_lens.long() - 1
        rows = torch.arange(2, device="cuda")
        cm.slot_mapping.copy_(cm.block_table_tensor[rows, positions // 16].long() * 16
                              + positions % 16)
        expected_cache = cache.clone()
        expected = torch.empty_like(q)
        impl.forward(layer, q, k, v, expected_cache, metadata, output=expected)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, expected, atol=0, rtol=0)
        torch.testing.assert_close(cache, expected_cache, atol=0, rtol=0)
