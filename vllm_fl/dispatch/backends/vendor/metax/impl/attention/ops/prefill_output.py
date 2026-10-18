# SPDX-License-Identifier: Apache-2.0
"""Pinned MetaX FA2 paged prefill destination ABI; no persistent tensor state."""
from collections import Counter
import os
import threading
import torch
import flash_attn
from flash_attn import flash_attn_interface as fa

_PUBLIC = fa.flash_attn_varlen_func
_FORWARD = fa._flash_attn_varlen_forward
_CONTIGUOUS = fa.maybe_contiguous
_NATIVE = fa.flash_attn_cuda.varlen_fwd
_AUTOGRAD = fa.FlashAttnVarlenFunc
_AUTOGRAD_FORWARD = _AUTOGRAD.forward
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
    begin = t.data_ptr()
    return begin, begin + (1 + sum((n-1)*s for n, s in zip(t.shape, t.stride()))) * t.element_size()


def try_prefill_into(public, q, k, v, output, table, cu_q, cu_k,
                     max_q, max_k, scale, causal, window, alibi, softcap, sinks):
    supported = (flash_attn.__version__ == '2.6.3' and public is _PUBLIC
                 and fa.flash_attn_varlen_func is _PUBLIC
                 and fa._flash_attn_varlen_forward is _FORWARD
                 and fa.maybe_contiguous is _CONTIGUOUS
                 and fa.flash_attn_cuda.varlen_fwd is _NATIVE
                 and fa.FlashAttnVarlenFunc is _AUTOGRAD
                 and _AUTOGRAD.forward is _AUTOGRAD_FORWARD
                 and 'MHA_INPUT_COLLECTION' not in os.environ
                 and not torch.is_grad_enabled()
                 and alibi is None and sinks is None and softcap == 0.0
                 and window == (-1, -1) and type(causal) is bool
                 and type(max_q) is int and type(max_k) is int
                 # The backend routes <=128-query requests through its decode path.
                 # Keep small varlen/GQA layouts on the public fallback.
                 and 128 < max_q <= 8192 and 0 < max_k <= 16384
                 and all(type(t) is torch.Tensor for t in (q, k, v, output, table, cu_q, cu_k)))
    if supported:
        supported = (q.ndim == 3 and 0 < q.shape[0] <= 8192
                     and tuple(q.shape[1:]) == (16, 128) and output.shape == q.shape
                     and k.ndim == 4 and k.shape[0] > 0 and tuple(k.shape[1:]) == (16, 2, 128)
                     and v.shape == k.shape and table.ndim == 2
                     and 1 <= table.shape[0] <= 8 and table.shape[1] > 0
                     and cu_q.shape == cu_k.shape == (table.shape[0]+1,)
                     and q.device.type == 'cuda'
                     and all(t.device == q.device for t in (k, v, output, table, cu_q, cu_k))
                     and all(t.dtype == torch.bfloat16 and not t.requires_grad for t in (q, k, v, output))
                     and all(t.dtype == torch.int32 for t in (table, cu_q, cu_k))
                     and all(t.is_contiguous() for t in (k, v, output, cu_q, cu_k))
                     and q.stride(-1) == table.stride(-1) == 1
                     and all(s > 0 for t in (q, table) for s in t.stride()))
    if supported:
        begin, end = _interval(output)
        for t in (q, k, v, table, cu_q, cu_k):
            a, b = _interval(t)
            if not (end <= a or begin >= b):
                supported = False
                break
    if not supported:
        _count('fallback')
        return False
    # Exact installed _flash_attn_varlen_forward positional ABI, out at 3.
    # q/k/v have last-dimension stride 1, so maybe_contiguous is a no-op.
    # No catch: after native submission an exception must never rerun fallback.
    results = fa.flash_attn_cuda.varlen_fwd(
        q, k, v, output, cu_q, cu_k, None, None, table, None,
        max_q, max_k, 0.0, scale, False, causal, -1, -1, 0.0,
        False, None, None, False)
    result = results[0]
    if (result.data_ptr() != begin or result.shape != output.shape
            or result.stride() != output.stride()):
        raise RuntimeError('pinned FA2 prefill output did not return destination storage/layout')
    _count('hit')
    return True
