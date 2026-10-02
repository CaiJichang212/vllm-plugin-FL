# Copyright (c) 2026 BAAI. All rights reserved.
"""Narrow paged-prefill specialization of the installed vLLM Triton backend.

KV updates, metadata construction, cache layout and pure decode remain owned by
vLLM. Unsupported cases fall back before launching any candidate GPU operation.
Kernel/compilation errors deliberately propagate; a failed device is not reused.
"""
from __future__ import annotations

import math

import torch
from vllm import envs
from vllm.config import get_current_vllm_config_or_none
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
)


class IluvatarPagedPrefillBackend(TritonAttentionBackend):
    @staticmethod
    def get_impl_cls():
        return IluvatarPagedPrefillImpl


class IluvatarPagedPrefillImpl(TritonAttentionImpl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = get_current_vllm_config_or_none()
        parallel = None if config is None else config.parallel_config
        cache = None if config is None else config.cache_config
        self._paged_prefill_enabled = (
            getattr(current_platform, "vendor_name", None) == "iluvatar"
            and parallel is not None
            and getattr(parallel, "decode_context_parallel_size", 1) == 1
            and getattr(parallel, "prefill_context_parallel_size", 1) == 1
            and not getattr(cache, "kv_sharing_fast_prefill", False)
            and not envs.VLLM_BATCH_INVARIANT
            and self.attn_type == AttentionType.DECODER
            and (self.num_heads, self.num_kv_heads, self.head_size) == (16, 2, 128)
            and math.isfinite(self.scale)
            and self.kv_cache_dtype in ("auto", "bfloat16")
            and not self._is_per_token_head_quant
            and self.alibi_slopes is None
            and not self.use_alibi_sqrt
            and self.sliding_window == (-1, -1)
            and self.logits_soft_cap == 0
            and self.sinks is None
            and self.chunk_lookback == -1
            and self.kv_sharing_target_layer_name is None
            and not self.use_td
        )
        # Resolve once, outside forward/Graph capture. Missing candidate code must
        # fail initialization, not silently masquerade as an enabled optimization.
        self._paged_prefill = None
        if self._paged_prefill_enabled:
            from flag_gems.runtime.backend._iluvatar.fused.paged_prefill_attention import (
                paged_prefill_attention,
            )
            self._paged_prefill = paged_prefill_attention

    def _supports(self, query, kv_cache, metadata, output, output_scale, output_block_scale):
        if (not self._paged_prefill_enabled or metadata is None
                or output_scale is not None or output_block_scale is not None):
            return False
        # Identity check rejects per-request device boolean tensors without reading them.
        if (getattr(metadata, "causal", None) is not True
                or getattr(metadata, "use_cascade", None) is not False
                or getattr(metadata, "mm_prefix_range", None) is not None
                or getattr(metadata, "mm_prefix_range_tensor", None) is not None):
            return False
        max_query = getattr(metadata, "max_query_len", None)
        actual = getattr(metadata, "num_actual_tokens", None)
        if (type(max_query) is not int or max_query <= 1
                or type(actual) is not int or actual <= 0):
            return False
        if (query.ndim != 3 or output.ndim != 3 or kv_cache.ndim != 5
                or tuple(query.shape[1:]) != (16, 128)
                or tuple(output.shape) != tuple(query.shape)
                or kv_cache.shape[0] <= 0 or kv_cache.shape[1] != 2
                or kv_cache.shape[2] not in (16, 32, 64)
                or tuple(kv_cache.shape[3:]) != (2, 128)
                or actual > query.shape[0] or actual > output.shape[0]
                or max_query > actual):
            return False
        if query.device.type != "cuda":
            return False
        for tensor in (query, kv_cache, output):
            if (tensor.dtype != torch.bfloat16 or tensor.device != query.device
                    or any(stride <= 0 for stride in tensor.stride())):
                return False
        starts = getattr(metadata, "query_start_loc", None)
        lengths = getattr(metadata, "seq_lens", None)
        table = getattr(metadata, "block_table", None)
        if starts is None or lengths is None or table is None:
            return False
        if (starts.ndim != 1 or lengths.ndim != 1 or table.ndim != 2
                or lengths.shape[0] <= 0 or starts.shape[0] != lengths.shape[0] + 1
                or table.shape[0] != lengths.shape[0] or table.shape[1] <= 0):
            return False
        max_seq = getattr(metadata, "max_seq_len", None)
        if (type(max_seq) is not int or max_seq <= 0
                or max_seq > table.shape[1] * kv_cache.shape[2]):
            return False
        for tensor in (starts, lengths, table):
            if (tensor.device != query.device or tensor.dtype not in (torch.int32, torch.int64)
                    or any(stride <= 0 for stride in tensor.stride())):
                return False
        return True

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output,
                output_scale=None, output_block_scale=None):
        if not self._supports(query, kv_cache, attn_metadata, output,
                              output_scale, output_block_scale):
            return super().forward(layer, query, key, value, kv_cache, attn_metadata, output,
                                   output_scale=output_scale, output_block_scale=output_block_scale)
        key_cache, value_cache = kv_cache.unbind(1)
        self._paged_prefill(query, key_cache, value_cache, output,
                            attn_metadata.query_start_loc, attn_metadata.seq_lens,
                            attn_metadata.block_table, attn_metadata.max_query_len, self.scale,
                            num_actual_tokens=attn_metadata.num_actual_tokens)
        return output
