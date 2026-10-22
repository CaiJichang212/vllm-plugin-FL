# SPDX-License-Identifier: Apache-2.0
"""Bind only an existing MiniCPM LlamaMLP instance; retain its complete fallback."""
from types import MethodType
from collections import Counter

import torch
from torch.nn.modules import module as module_hooks
from vllm import envs
from vllm.config import CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                              RowParallelLinear, UnquantizedLinearMethod)
from vllm.model_executor.models.llama import LlamaMLP
from vllm_fl.ops.activation import SiluAndMulFL


_observe = False
_counts = Counter()

def diagnostics(enabled=True):
    global _observe
    _observe = enabled
    return dict(_counts)


def _identity(module):
    forward = module.forward
    return (type(module), getattr(forward, '__func__', forward),
            getattr(module, '_forward_method', None), getattr(module, 'quant_method', None))


def _hooked(module):
    return any(bool(getattr(module, name, {})) for name in
               ('_forward_hooks', '_forward_pre_hooks', '_backward_hooks',
                '_backward_pre_hooks'))


def fallback_reason(mlp, x):
    saved = mlp._c500_ffn_saved
    if tuple(id(v) for v in (mlp.gate_up_proj, mlp.act_fn, mlp.down_proj)) != saved['ids']:
        return 'replaced_module'
    modules = (mlp.gate_up_proj, mlp.act_fn, mlp.down_proj)
    if tuple(_identity(v) for v in modules) != saved['identities']:
        return 'changed_dispatch'
    if any(_hooked(v) for v in modules):
        return 'submodule_hooks'
    if any(bool(getattr(module_hooks, k, {})) for k in
           ('_global_forward_hooks', '_global_forward_pre_hooks', '_global_backward_hooks',
            '_global_backward_pre_hooks')):
        return 'global_hooks'
    if torch.is_grad_enabled() or not torch.is_inference_mode_enabled():
        return 'autograd'
    if torch.compiler.is_compiling() or envs.VLLM_BATCH_INVARIANT:
        return 'compile_or_batch_invariant'
    if get_forward_context().cudagraph_runtime_mode != CUDAGraphMode.NONE:
        return 'graph'
    gate, _, down = modules
    if (type(gate.quant_method) is not UnquantizedLinearMethod
            or type(down.quant_method) is not UnquantizedLinearMethod
            or gate.bias is not None or down.bias is not None
            or gate.tp_size != 1 or down.tp_size != 1):
        return 'quant_bias_tp'
    if any(getattr(v, name, None) is not None for v in modules
           for name in ('lora_a', 'lora_b', 'adapter', 'adapter_layer', 'lora_manager')):
        return 'adapter'
    if tuple(gate.weight.shape) != (12288, 2048) or tuple(down.weight.shape) != (2048, 6144):
        return 'weight_layout'
    return None


def _forward(self, x):
    saved = self._c500_ffn_saved
    reason = fallback_reason(self, x)
    if reason is not None:
        if _observe: _counts['fallback:'+reason] += 1
        return saved['forward'](x)
    from flag_gems.runtime.backend._metax.fused.gate_up_silu import try_gate_up_silu
    out = try_gate_up_silu(x, self.gate_up_proj.weight)
    if out is None:
        if _observe: _counts['fallback:kernel_support'] += 1
        return saved['forward'](x)
    if _observe: _counts['fused:'+saved['prefix']] += 1
    from .minicpm5_down_kn import project_down_kn
    return project_down_kn(self, out)


def bind_gate_up_silu(mlp, config, prefix=''):
    # RopeKVDecoderLayer is instantiated only by the already narrow eager model gate.
    from flag_gems import runtime
    if runtime.device.vendor_name != 'metax':
        return False
    if (type(mlp) is not LlamaMLP or type(mlp.gate_up_proj) is not MergedColumnParallelLinear
            or type(mlp.down_proj) is not RowParallelLinear or type(mlp.act_fn) is not SiluAndMulFL
            or getattr(config, 'lora_config', None) is not None
            or config.quant_config is not None
            or not config.model_config.enforce_eager
            or config.parallel_config.tensor_parallel_size != 1
            or config.parallel_config.pipeline_parallel_size != 1):
        return False
    if '_c500_ffn_saved' in vars(mlp):
        return False
    modules = (mlp.gate_up_proj, mlp.act_fn, mlp.down_proj)
    mlp._c500_ffn_saved = dict(forward=mlp.forward, ids=tuple(map(id, modules)),
                              identities=tuple(map(_identity, modules)), prefix=prefix)
    mlp.forward = MethodType(_forward, mlp)
    return True
