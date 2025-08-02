# -- coding: utf-8 --
from typing import Union

import torch
import torch.nn as nn
from jaxtyping import Float

from models.transformer.attention import MultiHeadAttention, softmax
from models.transformer.embedding import Embedding
from models.transformer.ffn import PointWiseFFN
from models.transformer.linear import Linear
from models.transformer.rmsnorm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, use_causal_mask: bool = False, 
                 use_rope: bool = False, theta: Union[float, None] = None, max_seq_len: Union[int, None] = None,
                 token_positions: Union[torch.Tensor, None] = None, device=None, dtype=None):
        super(TransformerBlock, self).__init__()
        self.attention_prev_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.attention = MultiHeadAttention(d_model, num_heads, use_causal_mask, use_rope,
                                            theta, max_seq_len, token_positions, device, dtype)

        self.ffn_prev_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = PointWiseFFN(d_model, d_ff, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_prev_norm(x))
        return x + self.ffn(self.ffn_prev_norm(x))


class TransformerLM(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float,
                 num_layers: int, vocab_size: int, device=None, dtype=None):
        super(TransformerLM, self).__init__()
        self.input_embedding = Embedding(vocab_size, d_model, device, dtype)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, use_causal_mask=True,
                                 use_rope=True, theta=theta, max_seq_len=max_seq_len,
                                 device=device, dtype=dtype)
                for _ in range(num_layers)
            ]
        )

        self.post_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.output_ffn = Linear(d_model, vocab_size, device, dtype)

    def forward(self, x: Float[torch.Tensor, "... batch_size seq_len"]):
        x = self.input_embedding(x)

        for block in self.blocks:
            x = block(x)

        return self.output_ffn(self.post_norm(x))

