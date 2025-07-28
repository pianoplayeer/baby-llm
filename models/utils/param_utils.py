# -- coding: utf-8 --
import torch
import torch.nn as nn


def norm_init_params(dim_in, dim_out, mean, std, a, b, tensor_kwargs: dict) -> nn.Parameter:
    weights = torch.empty(dim_in, dim_out, **tensor_kwargs)
    nn.init.trunc_normal_(weights, mean, std, a, b)
    return nn.Parameter(weights)
