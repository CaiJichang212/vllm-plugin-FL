# SPDX-License-Identifier: Apache-2.0
"""Private decoder integration; public RMSNorm remains in-place."""
import torch
from vllm.model_executor.layers.attention.attention import get_attention_context
from .minicpm5_ffn_fusion import fallback_reason, _hooked, _identity


def snapshot(layer, prefix):
    modules=(layer.mlp,layer.post_attention_layernorm)
    if not hasattr(layer.mlp,'_c500_ffn_saved'):
        return
    layer._post_norm_saved=(modules,tuple(_identity(m) for m in modules),prefix)


def allowed(layer, x):
    saved=getattr(layer,'_post_norm_saved',None)
    if saved is None or type(x) is not torch.Tensor or tuple(x.shape)!=(4,2048):
        return False
    modules=(layer.mlp,layer.post_attention_layernorm)
    if any(a is not b for a,b in zip(modules,saved[0])) or tuple(_identity(m) for m in modules)!=saved[1]:
        return False
    if any(_hooked(m) for m in modules) or fallback_reason(layer.mlp,x) is not None:
        return False
    metadata,_,_,_=get_attention_context(saved[2]+'.self_attn.attn')
    return (metadata is not None and metadata.num_prefills==0 and metadata.num_prefill_tokens==0
            and metadata.num_decodes==4 and metadata.num_decode_tokens==4 and metadata.num_actual_tokens==4)


def project_post_norm(layer_module, x, residual):
    if not allowed(layer_module,x):
        return None
    from flag_gems.runtime.backend._metax.fused.post_norm_gate import try_post_norm_gate
    norm=layer_module.post_attention_layernorm
    parts=layer_module._post_norm_saved[2].split('.')
    layer=int(parts[-1]);phase='decode'
    result=try_post_norm_gate(x,residual,norm.weight,layer_module.mlp.gate_up_proj.weight,norm.variance_epsilon)
    if result is None:
        return None
    out,new_residual=result
    out,_=layer_module.mlp.down_proj(out)
    return out,new_residual
