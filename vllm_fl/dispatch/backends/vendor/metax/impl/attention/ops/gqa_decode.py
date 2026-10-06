# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental eager BF16 GQA decode; caller owns semantic dispatch/fallback.

One program shares paged K/V across eight query heads. Splits are tile-aligned;
page indices are loaded once per page rather than once per token. Each query
head independently reduces the small FP32 partial-state array in stage two.
Only ordinary causal single-token decode is supported (no bias/window/sinks).
Caller must guarantee valid block indices and 0 <= sequence lengths <= 8192.
No device-to-host reads or inference of sequence lengths occur here.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _partials(Q, K, V, BT, LENS, PART, LSE,
              QB: tl.constexpr, QH: tl.constexpr,
              KP: tl.constexpr, KT: tl.constexpr, KH: tl.constexpr,
              VP: tl.constexpr, VT: tl.constexpr, VH: tl.constexpr,
              BTS: tl.constexpr, LSTR: tl.constexpr,
              SCALE: tl.constexpr, SPLITS: tl.constexpr, N: tl.constexpr):
    batch = tl.program_id(0)
    kv_head = tl.program_id(1)
    split = tl.program_id(2)
    length = tl.load(LENS + batch * LSTR)
    tiles_per_split = tl.cdiv(tl.cdiv(length, N), SPLITS)
    start = split * tiles_per_split * N
    stop = tl.minimum(start + tiles_per_split * N, length)
    h = tl.arange(0, 16)
    d = tl.arange(0, 128)
    q = tl.load(Q + batch * QB + (kv_head * 8 + h[:, None]) * QH
                + d[None, :], mask=h[:, None] < 8, other=0)
    maximum = tl.full((16,), -float("inf"), tl.float32)
    denominator = tl.zeros((16,), tl.float32)
    accumulator = tl.zeros((16, 128), tl.float32)
    for base in range(start, stop, N):
        # All split/tile boundaries align with page16; broadcast page lookup.
        pages = base // 16 + tl.arange(0, N // 16)
        physical = tl.load(BT + batch * BTS + pages,
                           mask=pages * 16 < length, other=0)
        token = base + tl.arange(0, N)
        kp = tl.reshape(physical[:, None] * KP
                        + tl.arange(0, 16)[None, :] * KT, (N,))
        vp = tl.reshape(physical[:, None] * VP
                        + tl.arange(0, 16)[None, :] * VT, (N,))
        k = tl.load(K + kp[None, :] + kv_head * KH + d[:, None],
                    mask=token[None, :] < stop, other=0)
        v = tl.load(V + vp[:, None] + kv_head * VH + d[None, :],
                    mask=token[:, None] < stop, other=0)
        score = tl.dot(q, k) * SCALE
        score = tl.where(token[None, :] < stop, score, -float("inf"))
        new_max = tl.maximum(maximum, tl.max(score, axis=1))
        alpha = tl.exp(maximum - new_max)
        probability = tl.exp(score - new_max[:, None])
        accumulator = accumulator * alpha[:, None] + tl.dot(probability.to(v.dtype), v)
        denominator = denominator * alpha + tl.sum(probability, axis=1)
        maximum = new_max
    valid = denominator > 0
    normalized = accumulator / tl.where(valid, denominator, 1.0)[:, None]
    logsum = tl.where(valid, maximum + tl.log(tl.where(valid, denominator, 1.0)),
                      -float("inf"))
    index = ((batch * 16 + kv_head * 8 + h) * SPLITS + split)
    tl.store(PART + index[:, None] * 128 + d[None, :], normalized,
             mask=h[:, None] < 8)
    tl.store(LSE + index, logsum, mask=h < 8)


@triton.jit
def _merge(PART, LSE, OUT, SPLITS: tl.constexpr):
    head = tl.program_id(0)
    split = tl.arange(0, SPLITS)
    d = tl.arange(0, 128)
    lse = tl.load(LSE + head * SPLITS + split)
    maximum = tl.max(lse, axis=0)
    # Entirely empty sequences produce zero, including all-empty split sets.
    safe_max = tl.where(maximum == -float("inf"), 0.0, maximum)
    weights = tl.exp(lse - safe_max)
    partial = tl.load(PART + (head * SPLITS + split[:, None]) * 128 + d[None, :])
    total = tl.sum(weights, axis=0)
    output = tl.sum(partial * weights[:, None], axis=0) / tl.where(total > 0, total, 1.0)
    tl.store(OUT + head * 128 + d, output)


def paged_gqa_decode(q, k_cache, v_cache, block_table, seq_lens,
                     softmax_scale, *, num_splits=4, block_n=32):
    """Return [B,1,16,128]; allocations are intentionally included in timing.

    Supports B=1/4 and noncontiguous batch strides in Q. Cache shape is
    [pages,16,2,128], with contiguous head/dimension and NHD page strides.
    This prototype does not provide graph capture or a general dispatch API.
    """
    assert q.ndim == 4 and tuple(q.shape[1:]) == (1, 16, 128)
    assert q.shape[0] in (1, 4)
    assert q.dtype == k_cache.dtype == v_cache.dtype == torch.bfloat16
    assert q.device.type == "cuda"
    assert all(t.device == q.device for t in (k_cache, v_cache, block_table, seq_lens))
    assert q.stride(-1) == 1
    assert k_cache.ndim == v_cache.ndim == 4
    assert k_cache.shape == v_cache.shape and tuple(k_cache.shape[1:]) == (16, 2, 128)
    assert all(t.stride(-1) == 1 and t.stride(-2) == 128
               and t.stride(-3) == 256 and t.stride(0) >= 4096
               for t in (k_cache, v_cache))
    assert block_table.ndim == 2 and block_table.shape[0] == q.shape[0]
    assert block_table.stride(1) == 1
    assert seq_lens.ndim == 1 and seq_lens.shape[0] == q.shape[0]
    assert block_table.dtype in (torch.int32, torch.int64)
    assert seq_lens.dtype in (torch.int32, torch.int64)
    assert num_splits in (4, 8) and block_n in (16, 32, 64)
    batch = q.shape[0]
    partial = torch.empty((batch, 16, num_splits, 128), device=q.device, dtype=torch.float32)
    lse = torch.empty((batch, 16, num_splits), device=q.device, dtype=torch.float32)
    output = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    _partials[(batch, 2, num_splits)](
        q, k_cache, v_cache, block_table, seq_lens, partial, lse,
        q.stride(0), q.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        block_table.stride(0), seq_lens.stride(0), softmax_scale,
        num_splits, block_n, num_warps=4, num_stages=1)
    _merge[(batch * 16,)](partial, lse, output, num_splits,
                         num_warps=4, num_stages=1)
    return output
