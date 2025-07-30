# -- coding: utf-8 --
import math
from typing import Union

import torch
import torch.nn as nn
import numpy as np

from models.utils.param_utils import norm_init_params


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super(Linear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.device: Union[torch.device, None] = device
        self.dtype: Union[torch.dtype, None] = dtype
        self.weights = self._init_params()

    def _init_params(self) -> nn.Parameter:
        factory_kwargs = {'device': self.device, 'dtype': self.dtype}
        mean = 0
        std = np.sqrt(2 / (self.in_features + self.out_features))

        return norm_init_params(self.out_features, self.in_features, mean, std,
                                -3 * std, 3 * std, factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weights.T
