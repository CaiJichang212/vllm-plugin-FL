# SPDX-License-Identifier: Apache-2.0
"""Call-local prefix/split view for ordinary M4 decode; no storage cache."""
import torch


def original_query_view(query, tokens, decodes, reshape, reason):
    return reshape(query[:tokens], decodes)


def decode_query_view(query, tokens, decodes, reshape, phase, layer_name):
    # Retain the original path for subclasses, speculative and mixed batches.
    layer_index = -1
    if isinstance(layer_name, str) and layer_name.startswith('model.layers.'):
        parts = layer_name.split('.')
        if len(parts) > 2 and parts[2].isdigit():
            layer_index = int(parts[2])
    if (type(query) is torch.Tensor and phase == 'decode'
            and 0 <= layer_index < 42 and tokens == decodes == 4
            and query.ndim == 3 and query.shape[0] >= 4
            and tuple(query.shape[1:]) == (16, 128)
            and query.device.type == 'cuda' and query.dtype == torch.bfloat16
            and not query.requires_grad):
        stride = query.stride()
        if all(s > 0 for s in stride) and stride[2] == 1:
            # Splitting the prefix dimension introduces a singleton axis.
            # Its stride is immaterial but matches Tensor.view exactly.
            return query.as_strided((4, 1, 16, 128),
                                    (stride[0], 16 * stride[1], stride[1], stride[2]))
    reason = 'outside_ordinary_m4_decode'
    return original_query_view(query, tokens, decodes, reshape, reason)
