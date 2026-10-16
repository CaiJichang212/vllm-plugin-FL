# SPDX-License-Identifier: Apache-2.0
"""Pinned MetaX Triton launch descriptors; no tensor/pointer/stream retention."""
from collections import Counter, OrderedDict
import os
import threading
import torch
from vllm.triton_utils import triton

_local = threading.local()
_observe = False


def diagnostics(enabled=False):
    global _observe
    _observe = enabled
    return dict(getattr(_local, 'counts', {}))


def _count(name):
    if _observe:
        if not hasattr(_local, 'counts'):
            _local.counts = Counter()
        _local.counts[name] += 1


def _cache(device):
    pid = os.getpid()
    if getattr(_local, 'pid', None) != pid:
        _local.pid, _local.devices, _local.counts = pid, {}, Counter()
    return _local.devices.setdefault(device, OrderedDict())


def _tail(t):
    q, k, v, pos, cos, sin, kc, vc, slots = t
    return (*q.stride(), *k.stride(), *v.stride(), pos.stride(0),
            *cos.stride(), *sin.stride(), slots.stride(0),
            *kc.stride(), *vc.stride(), slots.numel(), kc.shape[0]*16,
            cos.shape[0])


def _supported(entry, tensors):
    # Dynamic hooks and debug compilation keep the original JIT semantics.
    runtime = getattr(getattr(triton, 'knobs', None), 'runtime', None)
    return (triton.__version__ == '3.0.0'
            and all(type(t) is torch.Tensor and not t.requires_grad for t in tensors)
            and type(entry).__name__ == 'JITFunction'
            and not entry.debug and not entry.pre_run_hooks
            and entry.launch_metadata is None
            and entry.cache_hook is None
            and runtime is not None
            and runtime.jit_cache_hook is None
            and runtime.jit_post_compile_hook is None
            and getattr(entry.run, '__func__', None) is triton.runtime.jit.JITFunction.run)


def _globals_valid(entry):
    missing = object()
    return all(values.get(name, missing) == old
               for (name, _), (old, values) in entry.used_global_vals.items())


def launch(entry, tensors, tokens, num_warps):
    """The original call launches once; hot calls use its standard runner."""
    def original():
        return entry[(tokens,)](*tensors, *_tail(tensors),
                               num_warps=num_warps, enable_fp_fusion=True)
    if not _supported(entry, tensors):
        _count('fallback')
        return original()
    device = tensors[0].device.index
    cache = _cache(device)
    # These metadata determine every constexpr, pointer specialization and option.
    signature = (entry, entry.fn, entry.src, entry.cache_key,
                 tuple((type(t), t.dtype, tuple(t.shape), tuple(t.stride()),
                        t.data_ptr() % 16 == 0) for t in tensors),
                 tokens, num_warps, True)
    record = cache.get(signature)
    if record is not None:
        key, compiled_cache, kernel, runner, binder, compile_fn, tl_module = record
        if (entry.cache.get(device) is compiled_cache
                and compiled_cache.get(key) is kernel and entry.binder is binder
                and entry.compile is compile_fn
                and entry.__globals__.get('tl') is tl_module
                and _globals_valid(entry)):
            # CompiledKernel.__getitem__ obtains the CURRENT stream and hooks.
            runner(*tensors)
            cache.move_to_end(signature)
            _count('hit')
            return kernel
        del cache[signature]
        _count('invalidated')
    _count('cold')
    kernel = original()
    if kernel is None or not _supported(entry, tensors) or not _globals_valid(entry):
        _count('uncached')
        return kernel
    args = (*tensors, *_tail(tensors))
    _, sig, const, _, excess = entry.binder(
        *args, num_warps=num_warps, enable_fp_fusion=True, debug=entry.debug)
    key = ''.join(sig) + str((const, excess))
    compiled_cache = entry.cache[device]
    if compiled_cache.get(key) is not kernel:
        _count('uncached')
        return kernel
    cache[signature] = (key, compiled_cache, kernel, kernel[(tokens, 1, 1)],
                        entry.binder, entry.compile, entry.__globals__.get('tl'))
    cache.move_to_end(signature)
    if len(cache) > 64:
        cache.popitem(last=False)
        _count('evicted')
    return kernel
