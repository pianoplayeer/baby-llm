# -- coding: utf-8 --
from typing import Union

import torch
import torch.nn as nn

from models.utils.param_utils import norm_init_params


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device: Union[torch.device, None] = device
        self.dtype: Union[torch.dtype, None] = dtype

        self.weights = self._init_params()

    def _init_params(self) -> nn.Parameter:
        factory_kwargs = {'device': self.device, 'dtype': self.dtype}
        mean, std = 0, 1
        return norm_init_params(self.num_embeddings, self.embedding_dim, mean, std,
                                -3 * std, 3 * std, factory_kwargs)

    def forward(self, x: torch.Tensor):
        '''
        forward propagation
        :param x: a batch of token sequence, [batch_size, seq_len]
        :return: a batch of embedding seq, [batch_size, seq_len, embedding_dim]
        '''
        batch_size, seq_len = x.shape
        embeddings = torch.empty(batch_size, seq_len, self.embedding_dim, device=self.device, dtype=self.dtype)

        for i, seq in enumerate(x):
            for j, token_id in enumerate(seq):
                embeddings[i][j] = self.weights[token_id]

        return embeddings