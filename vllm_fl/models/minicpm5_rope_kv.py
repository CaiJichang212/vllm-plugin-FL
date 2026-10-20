# SPDX-License-Identifier: Apache-2.0
"""Narrow eager MiniCPM5-2B RoPE/cache fusion through Llama layer injection."""
import json
import os
import time

import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.model_executor.layers.attention.kv_transfer_utils import maybe_transfer_kv_layer
from vllm.model_executor.models.llama import LlamaAttention, LlamaDecoderLayer, LlamaForCausalLM
from vllm.v1.attention.backend import AttentionType

from vllm_fl.dispatch.backends.vendor.metax.impl.attention.flash_attn import FlashAttentionImpl
from vllm_fl.dispatch.backends.vendor.metax.impl.attention.ops.rope_kv import fused_rope_cache

NUM_WARPS = 8


def model_supported(config):
    hf = config.model_config.hf_config
    return (getattr(hf, 'model_type', None) == 'llama'
        and getattr(hf, 'hidden_size', None) == 2048
        and getattr(hf, 'num_hidden_layers', None) == 42
        and getattr(hf, 'num_attention_heads', None) == 16
        and getattr(hf, 'num_key_value_heads', None) == 2
        and getattr(hf, 'vocab_size', None) == 130560
        and config.model_config.dtype == torch.bfloat16
        and config.model_config.enforce_eager
        and config.parallel_config.tensor_parallel_size == 1
        and config.parallel_config.pipeline_parallel_size == 1
        and config.quant_config is None)


def fallback_reason(attention, rotary, query, cache, metadata, graph_mode):
    impl = attention.impl
    if not isinstance(impl, FlashAttentionImpl): return 'backend'
    if torch.compiler.is_compiling() or graph_mode != CUDAGraphMode.NONE: return 'compile_or_graph'
    if query.device.type != 'cuda' or query.dtype != torch.bfloat16: return 'dtype_device'
    if (attention.num_heads, attention.num_kv_heads, attention.head_size) != (16,2,128): return 'head_layout'
    if not rotary.is_neox_style or rotary.rotary_dim != 128 or rotary.head_size != 128: return 'rotary'
    if impl.attn_type != AttentionType.DECODER: return 'attention_type'
    if impl.dcp_world_size != 1: return 'dcp'
    if impl.kv_sharing_target_layer_name is not None: return 'shared_kv'
    if attention.query_quant is not None or attention.calculate_kv_scales or impl.kv_cache_dtype != 'auto': return 'quantization'
    if metadata is None or not cache.numel(): return 'profiling'
    if metadata.use_cascade: return 'cascade'
    if (cache.ndim != 5 or tuple(cache.shape[2:]) != (16,2,128)
            or not cache.is_contiguous() or cache.dtype != torch.bfloat16): return 'cache_layout'
    if metadata.slot_mapping.numel() > query.shape[0]: return 'slots'
    return None


def trace(layer, path, reason=None):
    destination=os.environ.get('C500_ROPE_KV_TRACE')
    if destination:
        with open(destination,'a') as stream:
            stream.write(json.dumps(dict(wall=time.time(),layer=layer,path=path,reason=reason))+'\n')


@maybe_transfer_kv_layer
def fused_attention(positions, query, key, value, table, output, layer_name):
    """KV transfer wait precedes BOTH cache insertion and attention."""
    metadata, layer, cache, _ = get_attention_context(layer_name)
    cosine, sine = table.chunk(2,dim=-1)
    key_cache,value_cache=cache.unbind(0)
    fused_rope_cache(query,key,value,positions.flatten(),cosine,sine,
                     key_cache,value_cache,metadata.slot_mapping,num_warps=NUM_WARPS)
    layer.impl.forward(layer,query,key,value,cache,metadata,output=output,
                       cache_already_written=True, _cache_views=(key_cache, value_cache))
    trace(layer_name,'fused')


_qkv_observe = False
_qkv_counts = {}


def qkv_diagnostics(enabled=True):
    global _qkv_observe
    _qkv_observe = enabled
    return dict(_qkv_counts)


def _dispatch_identity(module):
    from .minicpm5_ffn_fusion import _identity
    method = getattr(module, 'quant_method', None)
    apply = getattr(method, 'apply', None)
    return (id(module), _identity(module), getattr(apply, '__func__', apply))


def qkv_fallback_reason(self, positions, hidden, metadata, cache):
    from .minicpm5_ffn_fusion import _hooked
    from torch.nn.modules import module as hooks
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    saved = getattr(self, '_qkv_fusion_saved', None)
    modules = (self.qkv_proj, self.rotary_emb)
    if saved is None or tuple(map(_dispatch_identity, modules)) != saved:
        return 'dispatch_changed'
    if any(_hooked(m) for m in modules) or any(getattr(hooks, n, {}) for n in
            ('_global_forward_hooks', '_global_forward_pre_hooks', '_global_backward_hooks', '_global_backward_pre_hooks')):
        return 'hooks'
    if (type(self.qkv_proj.quant_method) is not UnquantizedLinearMethod
            or self.qkv_proj.bias is not None or self.qkv_proj.tp_size != 1
            or any(getattr(self.qkv_proj, n, None) is not None for n in
                   ('lora_a', 'lora_b', 'adapter', 'adapter_layer', 'lora_manager'))):
        return 'projection'
    if torch.is_grad_enabled() or not torch.is_inference_mode_enabled(): return 'autograd'
    if metadata is None or metadata.num_prefills or metadata.num_prefill_tokens:
        return 'prefill_or_profile'
    if (metadata.num_decodes != hidden.shape[0] or metadata.num_decode_tokens != hidden.shape[0]
            or metadata.num_actual_tokens != hidden.shape[0] or not 1 <= hidden.shape[0] <= 4):
        return 'not_single_token_decode'
    reason = fallback_reason(self.attn, self.rotary_emb, hidden, cache, metadata,
                             get_forward_context().cudagraph_runtime_mode)
    if reason is not None: return reason
    if positions.ndim != 1: return 'positions'
    if os.environ.get('C500_ROPE_KV_FORCE_FALLBACK') == '1': return 'diagnostic_force'
    from flag_gems.runtime.backend._metax.fused.qkv_rope_cache import supported
    if not supported(hidden, self.qkv_proj.weight, positions, self.rotary_emb.cos_sin_cache, cache, metadata.slot_mapping):
        return 'kernel_support'
    return None


def original_kv_pair(cache, reason):
    # Keep the public dispatcher for unsupported objects/layouts.
    return cache.unbind(0)


def decode_kv_pair(cache, x, phase, layer_name):
    """Call-local BF16 KV views; no cached tensor, pointer, or metadata."""
    parts = layer_name.split('.') if isinstance(layer_name, str) else []
    layer = int(parts[2]) if (len(parts) > 2 and parts[:2] == ['model', 'layers']
                              and parts[2].isdigit()) else -1
    if (phase == 'decode' and 0 <= layer < 42
            and type(cache) is torch.Tensor and type(x) is torch.Tensor
            and tuple(x.shape) == (4, 2560)
            and cache.ndim == 5 and cache.shape[0] == 2
            and tuple(cache.shape[2:]) == (16, 2, 128) and cache.shape[1] > 0
            and cache.dtype == x.dtype == torch.bfloat16
            and cache.device.type == 'cuda' and cache.device == x.device
            and not cache.requires_grad and not x.requires_grad
            and all(s > 0 for s in cache.stride())):
        size = tuple(cache.shape[1:])
        strides = cache.stride()
        offset = cache.storage_offset()
        # Preserve arbitrary positive strides and the caller's storage offset.
        # This line is observed only after the supported-domain guards.
        key = cache.as_strided(size, strides[1:], offset)
        value = cache.as_strided(size, strides[1:], offset + strides[0])
        return key, value
    reason = 'outside_ordinary_m4_decode'
    return original_kv_pair(cache, reason)


@maybe_transfer_kv_layer
def fused_qkv_attention(self, positions, hidden_states, output, layer_name):
    # Wait for transfer BEFORE projection/cache writes, save after attention.
    metadata, layer, cache, _ = get_attention_context(layer_name)
    from flag_gems.runtime.backend._metax.fused.qkv_rope_cache import try_qkv_rope_cache
    qkv = try_qkv_rope_cache(hidden_states, self.qkv_proj.weight, positions,
                             self.rotary_emb.cos_sin_cache, cache, metadata.slot_mapping)
    if qkv is None: raise RuntimeError('QKV support changed after transfer wait; no fallback')
    q, k, v = qkv.split([2048, 256, 256], dim=-1)
    q = q.view(-1, 16, 128); k = k.view(-1, 2, 128); v = v.view(-1, 2, 128)
    layer.impl.forward(layer, q, k, v, cache, metadata, output=output,
                       cache_already_written=True, _cache_views=decode_kv_pair(
                           cache, qkv, 'decode' if metadata.num_prefills == 0 else 'mixed', layer_name))
    if _qkv_observe: _qkv_counts[layer_name] = _qkv_counts.get(layer_name, 0) + 1
    trace(layer_name, 'qkv_rope_cache')
    return True


class RopeKVAttention(LlamaAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from vllm.model_executor.layers.linear import QKVParallelLinear
        from vllm import envs
        if type(self.qkv_proj) is QKVParallelLinear and not envs.VLLM_BATCH_INVARIANT:
            self._qkv_fusion_saved = tuple(map(_dispatch_identity, (self.qkv_proj, self.rotary_emb)))

    def forward(self, positions, hidden_states):
        metadata, _, cache, _ = get_attention_context(self.attn.layer_name)
        if qkv_fallback_reason(self, positions, hidden_states, metadata, cache) is None:
            fused_output = torch.empty((hidden_states.shape[0], 16, 128),
                                       device=hidden_states.device, dtype=hidden_states.dtype)
            if fused_qkv_attention(self, positions, hidden_states, fused_output, self.attn.layer_name):
                output, _ = self.o_proj(fused_output.view(-1, self.q_size))
                return output
        qkv,_=self.qkv_proj(hidden_states)
        q,k,v=qkv.split([self.q_size,self.kv_size,self.kv_size],dim=-1)
        metadata,_,cache,_=get_attention_context(self.attn.layer_name)
        reason=fallback_reason(self.attn,self.rotary_emb,q,cache,metadata,
                               get_forward_context().cudagraph_runtime_mode)
        if os.environ.get('C500_ROPE_KV_FORCE_FALLBACK')=='1': reason='diagnostic_force'
        if reason is not None:
            trace(self.attn.layer_name,'separated',reason)
            q,k=self.rotary_emb(positions,q,k)
            attention_output=self.attn(q,k,v)
        else:
            q=q.view(-1,16,128);k=k.view(-1,2,128);v=v.view(-1,2,128)
            output=torch.empty_like(q)
            table=self.rotary_emb.cos_sin_cache.to(positions.device)
            fused_attention(positions,q,k,v,table,output,self.attn.layer_name)
            attention_output=output.view(-1,self.q_size)
        output,_=self.o_proj(attention_output)
        return output


class RopeKVDecoderLayer(LlamaDecoderLayer):
    def __init__(self,vllm_config,prefix=''):
        super().__init__(vllm_config=vllm_config,prefix=prefix,attn_layer_type=RopeKVAttention)
        from .minicpm5_ffn_fusion import bind_gate_up_silu
        bind_gate_up_silu(self.mlp, vllm_config, prefix=prefix+'.mlp')


class MiniCPM5RopeKVForCausalLM(LlamaForCausalLM):
    def __init__(self,*,vllm_config,prefix=''):
        layer_type=RopeKVDecoderLayer if model_supported(vllm_config) else LlamaDecoderLayer
        super().__init__(vllm_config=vllm_config,prefix=prefix,layer_type=layer_type)
