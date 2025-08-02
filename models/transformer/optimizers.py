# -- coding: utf-8 --
import math
from typing import Optional, Callable, overload, Any, Iterable

import torch
import torch.nn as nn
from torch.optim.optimizer import ParamsT


@torch.no_grad()
def grad_clip(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps=1e-6):
    total_norm = 0.0

    for param in parameters:
        if param.grad is not None:
            total_norm += (param.grad ** 2).sum()
    total_norm = math.sqrt(total_norm)

    clip_factor = max_l2_norm / (eps + total_norm)
    if clip_factor < 1:
        for param in parameters:
            if param.grad is not None:
                param.grad *= clip_factor


def cos_lr_schedule(t: int, lr_max: float, lr_min: float, t_w: int, t_c: int) -> float:
    if t < t_w:
        return lr_max * t / t_w
    if t_w <= t < t_c:
        return lr_min + 0.5 * (1 + math.cos(math.pi * (t - t_w) / (t_c - t_w))) * (lr_max - lr_min)
    return lr_min


class AdamW(torch.optim.Optimizer):
    def __init__(self, params: ParamsT, lr=1e-3, betas: tuple[float, float] = (0.9, 0.99), eps=1e-6, weight_decay=0.01):
        if lr < 0:
            raise ValueError('learning rate cannot be negative')

        defaults = {'lr': lr, 'm_beta': betas[0], 'v_beta': betas[1], 'eps': eps, 'weight_decay': weight_decay}
        super(AdamW, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            lr = group['lr']
            m_beta, v_beta = group['m_beta'], group['v_beta']
            weight_decay = group['weight_decay']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get('t', 1)

                m, v = state.get('m', torch.zeros_like(p)), state.get('v', torch.zeros_like(p))
                m = m_beta * m + (1 - m_beta) * p.grad
                v = v_beta * v + (1 - v_beta) * p.grad ** 2
                lr_t = lr * math.sqrt(1 - v_beta ** t) / (1 - m_beta ** t)
                p -= lr_t * m / (eps + torch.sqrt(v))
                p -= lr * weight_decay * p

                state['t'] = t + 1
                state['m'] = m
                state['v'] = v

        return loss


class SGD(torch.optim.Optimizer):
    def __init__(self, params: ParamsT, lr=1e-3):
        if lr < 0:
            raise ValueError('learning rate cannot be negative')

        defaults = {'lr': lr}
        super(SGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            lr = group['lr']
            for param in group['params']:
                grad = param.grad
                if grad is None:
                    continue

                state = self.state[param]
                t = state.get('t', 0)
                param -= lr / math.sqrt(t + 1) * grad
                state['t'] = t + 1

        return loss


if __name__ == '__main__':
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = AdamW([weights], lr=1)
    for t in range(100):
        opt.zero_grad()  # Reset the gradients for all learnable parameters.
        loss = (weights ** 2).mean()  # Compute a scalar loss value.
        print(loss.cpu().item())
        loss.backward()  # Run backward pass, which computes gradients.
        opt.step()  # Run optimizer step.
