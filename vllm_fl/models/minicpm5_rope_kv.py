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


def _qkv_head_views(qkv):
    """Return alias-only head views, or None to retain the original layout path.

    A padded row stride and a nonzero storage offset are valid: only the
    contiguous final dimension is split. Singleton and empty inputs retain the original
    inferred-dimension behavior, and subclasses retain their own dispatch.
    """
    if (type(qkv) is not torch.Tensor or qkv.dtype != torch.bfloat16
            or qkv.ndim != 2 or qkv.shape[0] < 2 or qkv.shape[1] != 2560
            or qkv.stride(1) != 1 or qkv.stride(0) < 2560):
        return None
    return qkv.view(qkv.shape[0], 20, 128).split((16, 2, 2), dim=1)


class RopeKVAttention(LlamaAttention):
    def forward(self, positions, hidden_states):
        qkv,_=self.qkv_proj(hidden_states)
        views = _qkv_head_views(qkv) if (self.q_size, self.kv_size) == (2048, 256) else None
        if views is None:
            q,k,v=qkv.split([self.q_size,self.kv_size,self.kv_size],dim=-1)
        else:
            q,k,v=views
        metadata,_,cache,_=get_attention_context(self.attn.layer_name)
        reason=fallback_reason(self.attn,self.rotary_emb,q,cache,metadata,
                               get_forward_context().cudagraph_runtime_mode)
        if os.environ.get('C500_ROPE_KV_FORCE_FALLBACK')=='1': reason='diagnostic_force'
        if reason is not None:
            if views is not None:
                q,k,v=qkv.split([self.q_size,self.kv_size,self.kv_size],dim=-1)
            trace(self.attn.layer_name,'separated',reason)
            q,k=self.rotary_emb(positions,q,k)
            attention_output=self.attn(q,k,v)
        else:
            if views is None:
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
