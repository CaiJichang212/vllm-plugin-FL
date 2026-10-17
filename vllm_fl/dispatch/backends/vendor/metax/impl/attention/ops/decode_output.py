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


def _interval(t):
    # Only positive-stride supported tensors reach this check.
    return t.data_ptr(), t.data_ptr() + (1 + sum((n-1)*s for n, s in zip(t.shape, t.stride()))) * t.element_size()


def try_decode_into(public, q, k_cache, v_cache, output, block_table,
                    cache_seqlens, scale, window, alibi, softcap, sinks):
    supported = (flash_attn.__version__ == '2.6.3' and public is _PUBLIC
                 and fa.flash_attn_with_kvcache is _PUBLIC
                 and fa.maybe_contiguous is _CONTIGUOUS
                 and 'MHA_INPUT_COLLECTION' not in os.environ
                 and alibi is None and sinks is None and softcap == 0.0
                 and window == (-1, -1)
                 and all(type(t) is torch.Tensor for t in
                         (q, k_cache, v_cache, output, block_table, cache_seqlens)))
    if supported:
        supported = (q.ndim == 4 and 1 <= q.shape[0] <= 8 and tuple(q.shape[1:]) == (1, 16, 128)
                     and tuple(output.shape) == (q.shape[0], 16, 128)
                     and k_cache.ndim == 4 and tuple(k_cache.shape[1:]) == (16, 2, 128)
                     and k_cache.shape[0] > 0 and v_cache.shape == k_cache.shape
                     and block_table.ndim == 2 and block_table.shape[0] == q.shape[0]
                     and cache_seqlens.shape == (q.shape[0],)
                     and q.device.type == 'cuda'
                     and all(t.device == q.device for t in (k_cache, v_cache, output, block_table, cache_seqlens))
                     and all(t.dtype == torch.bfloat16 and not t.requires_grad for t in (q, k_cache, v_cache, output))
                     and block_table.dtype == cache_seqlens.dtype == torch.int32
                     and output.is_contiguous() and k_cache.is_contiguous() and v_cache.is_contiguous()
                     and q.stride(-1) == block_table.stride(-1) == cache_seqlens.stride(-1) == 1
                     and all(s > 0 for t in (q, block_table, cache_seqlens) for s in t.stride()))
    if supported:
        begin, end = _interval(output)
        supported = all(end <= a or begin >= b for a, b in
                        (_interval(t) for t in (q, k_cache, v_cache, block_table, cache_seqlens)))
    if not supported:
        _count('fallback')
        return False
    # The pinned C++ entry folds eight Q heads into seqlen before checking
    # out.shape. Give it the corresponding strided view of ordinary head
    # order, so restoring [B,1,16,128] is a view rather than a result copy.
    destination = output.view(q.shape[0], 2, 8, 128).transpose(1, 2)
    # Positional ABI matches this installed FA2 wrapper, including out at13.
    # k/v=None: this call only reads the already updated KV cache.
    result, _ = fa.flash_attn_cuda.fwd_kvcache(
        q, k_cache, v_cache, None, None, cache_seqlens,
        None, None, None, None, block_table, None, destination,
        scale, True, -1, -1, 0.0, True, 0, None)
    if result.data_ptr() != output.data_ptr():
        raise RuntimeError('pinned FA2 optional output did not return destination storage')
    _count('hit')
    return True
