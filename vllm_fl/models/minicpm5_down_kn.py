# SPDX-License-Identifier: Apache-2.0
"""Repack before vLLM memory profiling; never repack in timed forward calls."""
from types import MethodType
import torch
from vllm.model_executor.layers.attention.attention import get_attention_context
from .minicpm5_ffn_fusion import fallback_reason


def bind_down_kn(mlp):
    if not hasattr(mlp,'_c500_ffn_saved'):
        return False
    down=mlp.down_proj
    method=down.quant_method
    if hasattr(method,'_c500_kn_original'):
        return False
    original=method.process_weights_after_loading
    method._c500_kn_original=original
    down.register_buffer('_c500_kn_weight',None,persistent=False)
    down._c500_kn_record=None
    def post_load(self,layer):
        # Invalidate BEFORE reload processing; never retain a stale copy on error.
        layer._c500_kn_record=None
        layer._c500_kn_weight=None
        original(layer)
        from flag_gems.runtime.backend._metax.fused.down_weight_kn import pack_down_weight
        record=pack_down_weight(layer.weight)
        if record is not None:
            layer._c500_kn_weight=record[0]
            layer._c500_kn_record=record
    method.process_weights_after_loading=MethodType(post_load,method)
    down._c500_kn_post_load=method.process_weights_after_loading
    return True


def project_down_kn(mlp,x):
    down=mlp.down_proj
    prefix=mlp._c500_ffn_saved['prefix']
    layer=int(prefix.split('.')[-2]);phase='outside_domain'
    record=getattr(down,'_c500_kn_record',None)
    if (record is not None and type(x) is torch.Tensor and tuple(x.shape)==(4,6144)
            and fallback_reason(mlp,x) is None
            and down.quant_method.process_weights_after_loading==down._c500_kn_post_load
            and down._c500_kn_weight is record[0]):
        metadata,_,_,_=get_attention_context(prefix[:-3]+'self_attn.attn')
        if (metadata is not None and metadata.num_prefills==0 and metadata.num_prefill_tokens==0
                and metadata.num_decodes==4 and metadata.num_decode_tokens==4 and metadata.num_actual_tokens==4):
            phase='decode'
            from flag_gems.runtime.backend._metax.fused.down_weight_kn import try_down_kn
            out=try_down_kn(x,down.weight,record)
            if out is not None:
                return out
    out,_=down(x)
    return out
