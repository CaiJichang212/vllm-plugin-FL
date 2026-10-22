# SPDX-License-Identifier: Apache-2.0
"""Repack before vLLM memory profiling; never repack in timed forward calls."""
from types import MethodType


def invalidate(down):
    record=getattr(down,'_c500_kn_record',None)
    if record is not None:
        record[3]=False
    down._c500_kn_record=None
    down._c500_kn_weight=None


def install_loader_guard(down):
    import weakref
    weight=down.weight
    saved=getattr(down,'_c500_kn_loader_weight',None)
    if saved is not None and saved() is weight and weight.weight_loader is down._c500_kn_loader:
        return True
    original=getattr(weight,'weight_loader',None)
    if not callable(original):
        return False
    def load(param,*args,**kwargs):
        # vLLM loaders use param.data.copy_, which need not bump param._version.
        # Invalidate retained descriptors BEFORE such an update, including errors.
        invalidate(down)
        return original(param,*args,**kwargs)
    weight.weight_loader=load
    down._c500_kn_loader=load
    down._c500_kn_loader_weight=weakref.ref(weight)
    return True


def bind_down_kn(mlp):
    if not hasattr(mlp,'_c500_ffn_saved'):
        return False
    down=mlp.down_proj
    method=down.quant_method
    if hasattr(method,'_c500_kn_original'):
        return False
    original=method.process_weights_after_loading
    down.register_buffer('_c500_kn_weight',None,persistent=False)
    down._c500_kn_record=None
    apply=method.apply
    down._c500_kn_apply=getattr(apply,'__func__',apply)
    if not install_loader_guard(down):
        return False
    method._c500_kn_original=original
    def post_load(self,layer):
        # Invalidate BEFORE reload processing; never retain a stale copy on error.
        invalidate(layer)
        original(layer)
        if not install_loader_guard(layer):
            return
        from flag_gems.runtime.backend._metax.fused.down_weight_kn import pack_down_weight
        record=pack_down_weight(layer.weight)
        if record is not None:
            layer._c500_kn_weight=record[0]
            layer._c500_kn_record=record
    method.process_weights_after_loading=MethodType(post_load,method)
    down._c500_kn_post_load=method.process_weights_after_loading
    return True
