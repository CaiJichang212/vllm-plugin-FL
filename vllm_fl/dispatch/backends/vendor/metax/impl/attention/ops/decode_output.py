# SPDX-License-Identifier: Apache-2.0
"""Use the pinned MetaX FA2 destination ABI for ordinary BF16 decode.

The native GQA preparation, attention and split/combine kernels are unchanged.
Only the final copy into vLLM's already allocated output is removed. There is
no tensor, pointer, stream or plan cache. Unsupported calls use the public API.
"""
from collections import Counter
import os
import threading

import torch
import flash_attn
from flash_attn import flash_attn_interface as fa

_PUBLIC = fa.flash_attn_with_kvcache
_CONTIGUOUS = fa.maybe_contiguous
_local = threading.local()
_observe = False


def diagnostics(enabled=False):
    global _observe
    _observe = enabled
    return dict(getattr(_local, 'counts', {}))


def _count(label):
    if _observe:
        if not hasattr(_local, 'counts'):
            _local.counts = Counter()
        _local.counts[label] += 1


def _interval(t, shape, stride=None):
    # Positive strides are validated before aliases. Reuse metadata only within
    # this call; data_ptr already includes any nonzero storage offset.
    begin = t.data_ptr()
    if stride is None:
        stride = t.stride()
    return begin, begin + (1 + sum((n-1)*s for n, s in zip(shape, stride))) * t.element_size()


def _disjoint(begin, end, t, shape, stride=None):
    a, b = _interval(t, shape, stride)
    return end <= a or begin >= b


def _generic_disjoint(begin, end, q, q_shape, q_stride, k_cache, k_shape,
                      v_cache, v_shape, block_table, table_shape, table_stride,
                      cache_seqlens, lens_shape, lens_stride):
    return (_disjoint(begin, end, q, q_shape, q_stride)
            and _disjoint(begin, end, k_cache, k_shape)
            and _disjoint(begin, end, v_cache, v_shape)
            and _disjoint(begin, end, block_table, table_shape, table_stride)
            and _disjoint(begin, end, cache_seqlens, lens_shape, lens_stride))


def _m4_disjoint(begin, end, q, q_stride, k_cache, v_cache, blocks,
                 block_table, table_shape, table_stride, cache_seqlens, lens_stride):
    """Closed-form bounding intervals AFTER the ordinary layout/type guards.

    BF16 q=[4,1,16,128], contiguous BF16 KV=[blocks,16,2,128],
    int32 table=[4,width], int32 lengths=[4]; all strides positive.
    data_ptr includes storage_offset. No pointer, tensor or proof is cached.
    Bounding intervals remain conservative for padding, as in the reference.
    """
    q_begin = q.data_ptr()
    if not (end <= q_begin or begin >= q_begin + 2 * (3*q_stride[0] + 15*q_stride[2] + 128)):
        return False
    k_begin = k_cache.data_ptr()
    if not (end <= k_begin or begin >= k_begin + blocks*8192):
        return False
    v_begin = v_cache.data_ptr()
    if not (end <= v_begin or begin >= v_begin + blocks*8192):
        return False
    table_begin = block_table.data_ptr()
    if not (end <= table_begin or begin >= table_begin + 4*(3*table_stride[0] + table_shape[1])):
        return False
    lens_begin = cache_seqlens.data_ptr()
    return end <= lens_begin or begin >= lens_begin + 4*(3*lens_stride[0] + 1)


def try_decode_into(public, q, k_cache, v_cache, output, block_table,
                    cache_seqlens, scale, window, alibi, softcap, sinks,
                    *, phase=None, layer_name=None):
    supported = (flash_attn.__version__ == '2.6.3' and public is _PUBLIC
                 and fa.flash_attn_with_kvcache is _PUBLIC
                 and fa.maybe_contiguous is _CONTIGUOUS
                 and 'MHA_INPUT_COLLECTION' not in os.environ
                 and alibi is None and sinks is None and softcap == 0.0
                 and window == (-1, -1)
                 and all(type(t) is torch.Tensor for t in
                         (q, k_cache, v_cache, output, block_table, cache_seqlens)))
    if supported:
        # Assignment expressions retain the original short-circuit order.
        # Only ordinary Tensors reach here; nothing survives this invocation.
        supported = (q.ndim == 4 and 1 <= (q_shape := q.shape)[0] <= 8 and tuple(q_shape[1:]) == (1, 16, 128)
                     and tuple(output_shape := output.shape) == (q_shape[0], 16, 128)
                     and k_cache.ndim == 4 and tuple((k_shape := k_cache.shape)[1:]) == (16, 2, 128)
                     and k_shape[0] > 0 and (v_shape := v_cache.shape) == k_shape
                     and block_table.ndim == 2 and (table_shape := block_table.shape)[0] == q_shape[0]
                     and (lens_shape := cache_seqlens.shape) == (q_shape[0],)
                     and (q_device := q.device).type == 'cuda'
                     and all(t.device == q_device for t in (k_cache, v_cache, output, block_table, cache_seqlens))
                     and all(t.dtype == torch.bfloat16 and not t.requires_grad for t in (q, k_cache, v_cache, output))
                     and block_table.dtype == cache_seqlens.dtype == torch.int32
                     and output.is_contiguous() and k_cache.is_contiguous() and v_cache.is_contiguous()
                     and (q_stride := q.stride())[-1] == (table_stride := block_table.stride())[-1] == (lens_stride := cache_seqlens.stride())[-1] == 1
                     and all(s > 0 for stride in (q_stride, table_stride, lens_stride) for s in stride))
    if supported:
        begin, end = _interval(output, output_shape)
        # Context only chooses a host proof; it never weakens the guards above.
        parts = layer_name.split('.') if isinstance(layer_name, str) else ()
        layer_index = (int(parts[2]) if len(parts) > 2 and parts[:2] == ['model', 'layers']
                       and parts[2].isdigit() else -1)
        if q_shape[0] == 4 and table_shape[1] > 0 and phase == 'decode' and 0 <= layer_index < 42:
            supported = _m4_disjoint(begin, end, q, q_stride, k_cache, v_cache, k_shape[0],
                                     block_table, table_shape, table_stride, cache_seqlens, lens_stride)
        else:
            reason = 'outside_m4_decode_layer_domain'
            supported = _generic_disjoint(begin, end, q, q_shape, q_stride, k_cache, k_shape,
                                          v_cache, v_shape, block_table, table_shape, table_stride,
                                          cache_seqlens, lens_shape, lens_stride)
    if not supported:
        _count('fallback')
        return False
    # The pinned C++ entry folds eight Q heads into seqlen before checking
    # out.shape. Give it the corresponding strided view of ordinary head
    # order, so restoring [B,1,16,128] is a view rather than a result copy.
    # The already validated contiguous destination has ordinary Q-head order.
    # Construct the identical folded view directly; retain its storage offset.
    destination = output.as_strided((q_shape[0], 8, 2, 128),
                                    (2048, 128, 1024, 1))
    # Positional ABI matches this installed FA2 wrapper, including out at13.
    # k/v=None: this call only reads the already updated KV cache.
    result, _ = fa.flash_attn_cuda.fwd_kvcache(
        q, k_cache, v_cache, None, None, cache_seqlens,
        None, None, None, None, block_table, None, destination,
        scale, True, -1, -1, 0.0, True, 0, None)
    if result.data_ptr() != begin:
        raise RuntimeError('pinned FA2 optional output did not return destination storage')
    _count('hit')
    return True
