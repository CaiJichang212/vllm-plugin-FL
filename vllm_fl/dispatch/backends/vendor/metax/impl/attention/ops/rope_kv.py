# SPDX-License-Identifier: Apache-2.0
"""Standalone experimental NeoX RoPE + NHD paged KV write, not dispatched.

Caller guarantees disjoint logical Q/K/V elements (shared QKV storage is fine),
no alias between inputs and caches, and unique nonnegative destination slots.
Positions must be in the supplied half-table range; nonnegative slots must be
within cache capacity. These data-dependent contracts are not host-synchronized.
Negative slots and tokens beyond slot_mapping are padding: Q still rotates,
but neither cache changes. K/V inputs stay unchanged; Q changes in place.
Only BF16, head_dim=128, Q heads=16, KV heads=2, page_size=16 are supported.
No graph/model integration, quantization, interleaving, or partial rotary.
"""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _rope_cache(Q, K, V, POS, COS, SIN, KC, VC, SLOT,
                QS: tl.constexpr, QH: tl.constexpr, QD: tl.constexpr,
                KS: tl.constexpr, KH: tl.constexpr, KD: tl.constexpr,
                VS: tl.constexpr, VH: tl.constexpr, VD: tl.constexpr,
                PS: tl.constexpr, CS: tl.constexpr, CD: tl.constexpr,
                SS: tl.constexpr, SD: tl.constexpr, MS: tl.constexpr,
                KCP: tl.constexpr, KCT: tl.constexpr,
                KCH: tl.constexpr, KCD: tl.constexpr,
                VCP: tl.constexpr, VCT: tl.constexpr,
                VCH: tl.constexpr, VCD: tl.constexpr,
                NSLOTS: tl.constexpr, CAPACITY: tl.constexpr,
                MAXPOS: tl.constexpr, SIN_OFFSET: tl.constexpr = 0):
    token = tl.program_id(0)
    pos = tl.load(POS + token * PS).to(tl.int64)
    valid_pos = (pos >= 0) & (pos < MAXPOS)
    tl.device_assert(valid_pos, "RoPE position out of range")
    d = tl.arange(0, 128)
    rotated = (d + 64) % 128
    cosine = tl.load(COS + pos * CS + (d % 64) * CD,
                     mask=valid_pos, other=0).to(tl.float32)
    sine = tl.load(SIN + SIN_OFFSET + pos * SS + (d % 64) * SD,
                   mask=valid_pos, other=0).to(tl.float32)
    sine = tl.where(rotated < 64, sine, -sine)
    # A complete token belongs to one CTA. Both halves of every Q head are
    # loaded before its store; there is no cross-CTA in-place dependency.
    hq = tl.arange(0, 16)
    qptr = Q + token * QS + hq[:, None] * QH
    q = tl.load(qptr + d[None, :] * QD)
    rq = tl.load(qptr + rotated[None, :] * QD)
    # Match FlagGems' FP32 table promotion and default FMA-enabled expression.
    out_q = q * cosine[None, :] + rq * sine[None, :]
    tl.store(qptr + d[None, :] * QD, out_q, mask=valid_pos)

    slot = tl.load(SLOT + token * MS, mask=token < NSLOTS,
                   other=-1).to(tl.int64)
    tl.device_assert(slot < CAPACITY, "KV slot out of range")
    active = (slot >= 0) & (slot < CAPACITY) & valid_pos
    # Sanitize padding addresses as well as masking every cache access.
    safe_slot = tl.where(active, slot, 0)
    page, offset = safe_slot // 16, safe_slot % 16
    hk = tl.arange(0, 2)
    kptr = K + token * KS + hk[:, None] * KH
    k = tl.load(kptr + d[None, :] * KD, mask=active, other=0)
    rk = tl.load(kptr + rotated[None, :] * KD, mask=active, other=0)
    out_k = k * cosine[None, :] + rk * sine[None, :]
    # Explicit BF16 rounding matches the standalone RoPE intermediate K.
    out_k = out_k.to(tl.bfloat16)
    v = tl.load(V + token * VS + hk[:, None] * VH + d[None, :] * VD,
                mask=active, other=0)
    tl.store(KC + page * KCP + offset * KCT + hk[:, None] * KCH
             + d[None, :] * KCD, out_k, mask=active)
    tl.store(VC + page * VCP + offset * VCT + hk[:, None] * VCH
             + d[None, :] * VCD, v, mask=active)


def _check_layout(tensor, name):
    # Accept ordinary padded/strided layouts, rejecting overlapping elements.
    span = 1
    for stride, size in sorted(zip(tensor.stride(), tensor.shape)):
        if size > 1:
            if stride < span:
                raise ValueError(f"{name} has an overlapping/unsupported layout")
            span += (size - 1) * stride


def _fused_rope_cache(q, k, v, positions, cos, sin, key_cache, value_cache,
                     slot_mapping, *, num_warps=4, whole_table=False):
    """Rotate Q in place and write rotated K / exact V to page16 NHD caches.

    Static validation only; see module docstring for caller-owned value and
    alias contracts. Supports strided half-tables, QKV views and cache views.
    Returns the original Q tensor. num_warps is explicitly 4 or 8; no autotune.
    """
    tensors = (q, k, v, positions, cos, sin, key_cache, value_cache, slot_mapping)
    if num_warps not in (4, 8):
        raise ValueError("num_warps must be 4 or 8")
    if any(t.device != q.device for t in tensors) or q.device.type != "cuda":
        raise ValueError("all tensors must be on the same MetaX CUDA device")
    if any(t.dtype != torch.bfloat16 for t in (q, k, v, key_cache, value_cache)):
        raise ValueError("Q/K/V and caches must be BF16")
    if q.ndim != 3 or q.shape[1:] != (16, 128):
        raise ValueError("Q must have shape [tokens,16,128]")
    tokens = q.shape[0]
    if k.shape != (tokens, 2, 128) or v.shape != k.shape:
        raise ValueError("K/V must have shape [tokens,2,128]")
    if (key_cache.ndim != 4 or key_cache.shape[1:] != (16, 2, 128)
            or value_cache.shape != key_cache.shape or key_cache.shape[0] == 0):
        raise ValueError("caches must have matching shape [pages,16,2,128]")
    if (cos.ndim != 2 or cos.shape[1] != (128 if whole_table else 64) or cos.shape[0] == 0
            or sin.shape != cos.shape
            or cos.dtype not in (torch.bfloat16, torch.float16, torch.float32)
            or sin.dtype != cos.dtype):
        raise ValueError("matching floating point half-tables [positions,64] required")
    if (positions.shape != (tokens,) or slot_mapping.ndim != 1
            or slot_mapping.numel() > tokens
            or positions.dtype not in (torch.int32, torch.int64)
            or slot_mapping.dtype not in (torch.int32, torch.int64)):
        raise ValueError("integer positions[tokens] and slots[<=tokens] required")
    for name, tensor in zip(
            ("q", "k", "v", "positions", "cos", "sin", "key_cache",
             "value_cache", "slot_mapping"), tensors):
        _check_layout(tensor, name)
    if tokens:
        with torch.cuda.device(q.device):
            _rope_cache[(tokens,)](
                q, k, v, positions, cos, sin, key_cache, value_cache, slot_mapping,
                *q.stride(), *k.stride(), *v.stride(), positions.stride(0),
                *cos.stride(), *sin.stride(), slot_mapping.stride(0),
                *key_cache.stride(), *value_cache.stride(),
                slot_mapping.numel(), key_cache.shape[0] * 16, cos.shape[0],
                SIN_OFFSET=64 * cos.stride(1) if whole_table else 0,
                num_warps=num_warps, enable_fp_fusion=True)
    return q


def fused_rope_cache(q, k, v, positions, cos, sin, key_cache, value_cache,
                     slot_mapping, *, num_warps=4):
    """Original split-table ABI; unchanged validation and zero sine offset."""
    return _fused_rope_cache(q, k, v, positions, cos, sin, key_cache,
                             value_cache, slot_mapping, num_warps=num_warps)


def _table_supported(table, *tensors):
    """Static whole-table domain; unsupported input falls back before launch.

    Tensor pointers already include storage_offset. Row/column strides describe
    the full [positions,128] table, including padding and nonzero offsets.
    """
    if any(type(t) is not torch.Tensor for t in (table, *tensors)):
        return False
    if (table.ndim != 2 or table.shape[1] != 128 or not table.shape[0]
            or table.dtype not in (torch.bfloat16, torch.float16, torch.float32)):
        return False
    try:
        _check_layout(table, 'table')
    except ValueError:
        return False
    return True


def _fused_rope_cache_table(q, k, v, positions, table, key_cache, value_cache,
                            slot_mapping, *, num_warps=8):
    """Private same-kernel path: sin begins64 column strides after cos."""
    if not _table_supported(table, q, k, v, positions, key_cache,
                            value_cache, slot_mapping):
        raise ValueError('unsupported whole RoPE table')
    return _fused_rope_cache(q, k, v, positions, table, table, key_cache,
                             value_cache, slot_mapping, num_warps=num_warps,
                             whole_table=True)
