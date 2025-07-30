# -- coding: utf-8 --
from typing import Union

import torch
import torch.nn as nn

from models.transformer.linear import Linear


class PointWiseFFN(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        '''
        (silu(x @ W1) * x @ W2) @ W3
        :param d_model:
        :param d_ff:
        :param device:
        :param dtype:
        '''
        super(PointWiseFFN, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device: Union[torch.device, None] = device
        self.dtype: Union[torch.dtype, None] = dtype

        self.w1: Linear = Linear(d_model, d_ff, device, dtype)
        self.w2: Linear = Linear(d_model, d_ff, device, dtype)
        self.w3: Linear = Linear(d_ff, d_model, device, dtype)

    def forward(self, x: torch.Tensor):
        '''
        :param x: [batch_size, seq_len, d_model]
        :return:
        '''
        return self.w3(silu(self.w1(x)) * self.w2(x))


def silu(x: torch.Tensor):
    return x * torch.sigmoid(x)
