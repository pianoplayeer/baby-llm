# -- coding: utf-8 --
import os
import pickle
from typing import BinaryIO, IO, Union

import numpy.typing as npt
import numpy as np
import torch

from models.tokenizer.bpe import BPETokenizer
from models.tokenizer.consts import BPEConstant
from models.utils.file_utils import read_file_to_chunks


def load_data(dataset: npt.NDArray, batch_size: int,
              context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    input_seqs, labels = [], []
    max_start_index = len(dataset) - context_length
    for _ in range(batch_size):
        i = np.random.randint(0, max_start_index)
        piece = dataset[i : i + context_length + 1]
        input_seqs.append(piece[:-1])
        labels.append(piece[1:])

    return torch.tensor(np.array(input_seqs), device=device), torch.tensor(np.array(labels), device=device)


def save_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    iteration: int,
                    out: Union[str, os.PathLike, BinaryIO, IO[bytes]]):
    states = {
        'model_state': model.state_dict(),
        'optim_state': optimizer.state_dict(),
        'iter': iteration
    }
    torch.save(states, out)


def load_checkpoint(src: Union[str, os.PathLike, BinaryIO, IO[bytes]],
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer) -> int:
    states = torch.load(src)
    model.load_state_dict(states['model_state'])
    optimizer.load_state_dict(states['optim_state'])

    return states['iter']


def save_bpe_tokenizer(tokenizer: BPETokenizer, path: Union[str, os.PathLike, BinaryIO, IO[bytes]]):
    data = {
        'vocab_size': tokenizer.vocab_size,
        'index2bytes': tokenizer.index2bytes,
        'merges': tokenizer.merges,
        'special_tokens': list(tokenizer.special_tokens),
    }

    with open(path, mode='wb') as f:
        pickle.dump(data, f)


def load_bpe_tokenizer(path: Union[str, BinaryIO]) -> BPETokenizer:
    with open(path, 'rb') if isinstance(path, str) else path as f:
        data = pickle.load(f)

    tokenizer = BPETokenizer(
        vocab_size=data['vocab_size'],
        special_tokens=data['special_tokens'],
    )
    tokenizer.index2bytes = data['index2bytes']
    tokenizer.bytes2index = {v: k for k, v in tokenizer.index2bytes.items()}
    tokenizer.merges = data['merges']

    return tokenizer


def transfer_dataset(tokenizer: BPETokenizer,
                     input_path: Union[str, BinaryIO],
                     output_path: Union[str, BinaryIO]):
    tokens = []
    chunks = read_file_to_chunks(input_path)

    for chunk in chunks:
        tokens += tokenizer.encode(chunk)

    np.save(output_path, np.array(tokens))


if __name__ == '__main__':
    a = 'hello world, nice to meet you,<|endoftext|> how are you'
    tokenizer = BPETokenizer(5, ['<|endoftext|>'])
    tokenizer.fit(a)

    indices = tokenizer.encode("worldto <|endoftext|>you")
    revert = tokenizer.decode(indices)

    print('origin: ' + "worldto <|endoftext|>you")
    print("encode: " + str(indices))
    print("decode: " + revert)

    tokenizer = load_bpe_tokenizer(BPEConstant.BPE_PATH)
    indices = tokenizer.encode("worldto <|endoftext|>you")
    revert = tokenizer.decode(indices)

    print('origin: ' + "worldto <|endoftext|>you")
    print("encode: " + str(indices))
    print("decode: " + revert)
