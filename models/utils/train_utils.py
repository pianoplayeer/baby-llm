# -- coding: utf-8 --
import os
from typing import Type, Union, Callable

import torch
import torch.nn as nn
import numpy as np
from loguru import logger
from matplotlib import pyplot as plt

from models.tokenizer.bpe import BPETokenizer
from models.transformer.optimizers import grad_clip
from models.utils.data_utils import load_data, save_checkpoint


def schedule_lr(optimizer: torch.optim.Optimizer, lr_scheduler: Callable[..., float],
                scheduler_params: dict):
    for group in optimizer.param_groups:
        group['lr'] = lr_scheduler(**scheduler_params)


def train_llm(tokenizer: BPETokenizer, train_chunks: list[str], test_chunks: list[str],
              model_type: Type[nn.Module], model_params: dict, optim_type: Type[torch.optim.Optimizer], optim_params: dict,
              loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
              lr_scheduler: Callable[..., float], scheduler_params: dict, max_l2_norm: float,
              batch_size: int, context_len: int, epochs: int, eps=1e-5, device=None) -> nn.Module:

    model = model_type(**model_params)
    model = torch.compile(model)
    optimizer = optim_type(model.parameters(), **optim_params)

    train_losses = []
    test_losses = []

    # train
    logger.info('start training loop')
    for t in range(1, epochs + 1):
        logger.info(f'available gpu mem: {torch.cuda.mem_get_info()[0]}')
        optimizer.zero_grad()
        chunk_idx = np.random.randint(0, len(train_chunks))
        indices = tokenizer.encode(train_chunks[chunk_idx])
        seqs, targets = load_data(np.array(indices), batch_size, context_len, device)

        pred = model(seqs)
        loss = loss_func(pred, targets)
        logger.info(f'epoch {t}, train loss: {loss}')
        train_losses.append(loss)
        loss.backward()
        grad_clip(model.parameters(), max_l2_norm, eps)
        scheduler_params['t'] = t
        schedule_lr(optimizer, lr_scheduler, scheduler_params)
        optimizer.step()

        # test
        chunk_idx = np.random.randint(0, len(test_chunks))
        indices = tokenizer.encode(test_chunks[chunk_idx])
        test_seqs, test_targets = load_data(np.array(indices), batch_size, context_len, device)

        with torch.no_grad():
            pred = model(test_seqs)
            loss = loss_func(pred, test_targets)
            test_losses.append(loss)
            logger.info(f'epoch {t}, test loss: {loss}')

        if t % 10 == 0 or t == epochs:
            save_checkpoint(model, optimizer, t, f'./model_checkpoint_{t}')

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, epochs + 1), test_losses, label='Test Loss', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_curve.png")
    plt.show()

    return model
