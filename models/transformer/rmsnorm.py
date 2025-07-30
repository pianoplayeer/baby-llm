# -- coding: utf-8 --
from typing import Union

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super(RMSNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps
        self.device: Union[torch.device, None] = device
        self.dtype: Union[torch.dtype, None] = dtype

        self.weights = self._init_params()

    def _init_params(self):
        factory_kwargs = {'device': self.device, 'dtype': self.dtype}

        weights = torch.ones(self.d_model, **factory_kwargs)
        return nn.Parameter(weights)

    def forward(self, x: torch.Tensor):
        in_type = x.dtype
        x.to(torch.float32)

        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        res = (x / rms) * self.weights
        res.to(in_type)

        return res
