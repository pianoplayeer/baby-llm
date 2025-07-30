# -- coding: utf-8 --
import math
from typing import Union

import einops
import torch
import torch.nn as nn
from jaxtyping import Float

from models.transformer.linear import Linear
from models.transformer.position_encode import RotaryPositionEncode
from models.utils.time_utils import time_cost


def softmax(x: torch.Tensor, dim: int):
    max_vals = x.max(dim=dim, keepdim=True).values
    x = torch.exp(x - max_vals)
    return x / x.sum(dim=dim, keepdim=True)


@time_cost
def scaled_dot_product_attention(
    q: Float[torch.Tensor, "... seq_len d_k"],
    k: Float[torch.Tensor, "... seq_len d_k"],
    v: Float[torch.Tensor, "... seq_len d_v"],
    mask: Float[torch.Tensor, "seq_len seq_len"]
):
    d_k = q.shape[-1]
    attention_scores = einops.einsum(q, k, "... q d_k, ... k d_k -> ... q k") / math.sqrt(d_k)

    if mask is not None:
        attention_scores = attention_scores.masked_fill(~mask, float('-inf'))
    return softmax(attention_scores, -1) @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, use_causal_mask: bool = False, use_rope: bool = False,
                 theta: Union[float, None] = None, max_seq_len: Union[int, None] = None,
                 token_positions: Union[torch.Tensor, None] = None, device=None, dtype=None):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_causal_mask = use_causal_mask
        self.use_rope = use_rope
        self.rope = RotaryPositionEncode(theta, d_model // num_heads, max_seq_len, device) if use_rope else None
        self.token_positions = token_positions

        self.w_q = Linear(d_model, d_model, device, dtype)
        self.w_k = Linear(d_model, d_model, device, dtype)
        self.w_v = Linear(d_model, d_model, device, dtype)
        self.w_o = Linear(d_model, d_model, device, dtype)

    def forward(self, x: Float[torch.Tensor, "... seq_len d_model"]):
        w_qkv = torch.concat([self.w_q.weights, self.w_k.weights, self.w_v.weights])
        qkv = x @ w_qkv.T
        q, k, v = qkv.chunk(3, -1)
        seq_len = x.shape[-2]

        q = einops.rearrange(q, "... seq_len (h dim) -> ... h seq_len dim", h=self.num_heads)
        k = einops.rearrange(k, "... seq_len (h dim) -> ... h seq_len dim", h=self.num_heads)
        v = einops.rearrange(v, "... seq_len (h dim) -> ... h seq_len dim", h=self.num_heads)

        if self.use_rope:
            q = self.rope(q, self.token_positions)
            k = self.rope(k, self.token_positions)

        causal_mask = None
        if self.use_causal_mask:
            causal_mask = torch.ones(seq_len, seq_len).tril(diagonal=0).bool().cuda()
        multi_heads = scaled_dot_product_attention(q, k, v, causal_mask)
        multi_heads = einops.rearrange(multi_heads, "... h seq_len dim -> ... seq_len (h dim)")

        return self.w_o(multi_heads)
