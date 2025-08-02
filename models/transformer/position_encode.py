# -- coding: utf-8 --
from typing import Union

import einops
import torch
import torch.nn as nn
from einops import rearrange


class RotaryPositionEncode(nn.Module):
    def __init__(self, theta: float, d_model: int, max_seq_len: int, device=None):
        '''
        add RoPE
        :param theta:
        :param d_model: need to be even
        :param max_seq_len:
        :param device:
        '''
        super(RotaryPositionEncode, self).__init__()
        self.theta = theta
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.device: Union[torch.device, None] = device

        self.rotate_matrix_table = self.gen_rotate_matrix()
        self.register_buffer('rotate_matrix', self.rotate_matrix_table, persistent=False)

    def gen_rotate_block(self, seq_pos, block_index) -> torch.Tensor:
        angle = torch.tensor(seq_pos / self.theta ** (2 * block_index / self.d_model))
        cos = torch.cos(angle)
        sin = torch.sin(angle)

        return torch.tensor([[cos, -sin], [sin, cos]], device=self.device)

    def gen_rotate_matrix(self) -> torch.Tensor:
        rotate_matrix = torch.zeros(self.max_seq_len, self.d_model, self.d_model, device=self.device)

        for pos in range(self.max_seq_len):
            mat = [self.gen_rotate_block(pos, k) for k in range(self.d_model // 2)]
            rotate_matrix[pos, :, :] = torch.block_diag(*mat)

        return rotate_matrix

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        *prev_dims, seq_len, d_model = x.shape
        if token_positions is None:
            token_positions = torch.arange(seq_len)

        rotate_mats = self.rotate_matrix_table[token_positions]
        x = x.to(device=self.device)

        x_rotated = rotate_mats @ x.unsqueeze(-1)
        x_rotated = x_rotated.squeeze(-1)
        return x_rotated
