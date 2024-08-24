# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn

from ..backends import LayerType, get_backend


class RMSNorm(nn.Module):
    """RMS Norm with add residual."""

    def __init__(self,
                 hidden_size: int,
                 eps: float = 1e-6,
                 dtype: torch.dtype = None,
                 device: torch.device = None,
                 is_w8a8: bool = False):
        super().__init__()
        backend = get_backend()
        if is_w8a8:
            builder = backend.get_layer_impl_builder(LayerType.RMSNormW8A8)
        else:
            builder = backend.get_layer_impl_builder(LayerType.RMSNorm)
        self.register_parameter('weight',
                                self.create_weight(hidden_size, dtype, device))
        self.impl = builder.build(hidden_size, eps)

    @staticmethod
    def create_weight(hidden_size: int,
                      dtype: torch.dtype = None,
                      device: torch.device = None):
        """create weight."""
        if dtype is None:
            dtype = torch.float16
        if device is None:
            device = 'cuda'
        weight = torch.nn.Parameter(torch.ones(hidden_size,
                                               dtype=dtype,
                                               device=device),
                                    requires_grad=False)
        return weight

    def forward(self, x: torch.Tensor, residual: torch.Tensor = None):
        """forward."""
        return self.impl.forward(x, self.weight, residual)
