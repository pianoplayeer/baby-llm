# -- coding: utf-8 --
import torch
from jaxtyping import Float


def cross_entropy_loss(output: Float[torch.Tensor, "... seq_len vocab_size"],
                       target: Float[torch.Tensor, "... seq_len"]):
    target_logits = output.gather(dim=-1, index=target.long().unsqueeze(-1))
    logits_sum = torch.logsumexp(output, dim=-1, keepdim=True)

    return torch.mean(-target_logits + logits_sum)


def perplexity(output: Float[torch.Tensor, "batch_size vocab_size"],
               target: Float[torch.Tensor, " batch_size"]):
    return torch.e ** cross_entropy_loss(output, target)
