# -- coding: utf-8 --
import os
import re
import time
from collections import defaultdict, Counter
from multiprocessing import Pool
from typing import Union, BinaryIO

import regex
from loguru import logger

from models.tokenizer.consts import BPEConstant
from models.utils.linkedlist import *
from models.utils.time_utils import time_cost


class BPETokenizer:
    # index -> bytes
    index2bytes: dict[int, bytes]

    bytes2index: dict[bytes, int]

    # merged byte pair -> index
    merges: dict[tuple[int, int], int]

    vocab_size: int

    special_tokens: set[str]

    specials_pattern: Union[str, None]

    def __init__(self, vocab_size, special_tokens: list[str] = None):
        self.vocab_size = vocab_size
        self.index2bytes = {x: bytes([x]) for x in range(BPEConstant.BASE_VOCAB_SIZE)}
        self.special_tokens = set()
        self.specials_pattern = None

        if special_tokens:
            self.special_tokens.update(special_tokens)
            self.specials_pattern = '|'.join(map(re.escape, special_tokens))
            for i, token in enumerate(self.special_tokens):
                self.index2bytes[BPEConstant.BASE_VOCAB_SIZE + i] = token.encode(encoding=BPEConstant.ENCODING)

        self.bytes2index = {v: k for k, v in self.index2bytes.items()}
        self.merges = {}

    def encode(self, text: str) -> list[int]:
        indices: list[int] = []
        split_text = [text]
        if self.specials_pattern:
            split_text = re.split('(' + self.specials_pattern + ')', text)

        for t in split_text:
            if t in self.special_tokens:
                indices.append(self.bytes2index[t.encode(encoding=BPEConstant.ENCODING)])
            else:
                cur_indices = list(map(int, t.encode(encoding=BPEConstant.ENCODING)))
                for pair, index in self.merges.items():
                    cur_indices = self._merge(cur_indices, pair, index)
                indices += cur_indices

        return indices

    def decode(self, indices: list[int]) -> str:
        bytes_list = list(map(self.index2bytes.get, indices))
        return b"".join(bytes_list).decode(encoding=BPEConstant.ENCODING)

    @time_cost
    def fit_file(self, input_path: Union[str, os.PathLike]):
        chunks = self._read_file_to_chunks(input_path)
        num_processes = min(len(chunks), os.cpu_count())

        with Pool(processes=num_processes) as pool:
            results = pool.map(self._split_by_patterns, chunks)
        tokens: list[list[int]] = [list(token) for token_list in results for token in token_list]
        self.do_fit(tokens)

    def _merge_pair(self, counter: defaultdict[tuple[int, int], int],
                    pos_index: defaultdict[tuple[int, int], set[int]],
                    tokens: list[list[int]], pair: tuple[int, int], new_index: int):

        for pos in pos_index[pair]:
            token_indices = tokens[pos]
            new_indices = []
            inner_pos_list = []

            i = 0
            while i < len(token_indices):
                if i + 1 < len(token_indices) \
                        and pair[0] == token_indices[i] \
                        and pair[1] == token_indices[i + 1]:
                    inner_pos_list.append(len(new_indices))
                    new_indices.append(new_index)

                    if i > 0 and (token_indices[i - 1], token_indices[i]) != pair:
                        counter[(token_indices[i - 1], pair[0])] -= 1

                    if i + 2 < len(token_indices) and (token_indices[i + 1], token_indices[i + 2]) != pair:
                        counter[(token_indices[i + 1], token_indices[i + 2])] -= 1

                    i += 2
                else:
                    new_indices.append(token_indices[i])
                    i += 1

            for j in range(len(inner_pos_list)):
                cur = inner_pos_list[j]
                if cur > 0 and (j == 0 or inner_pos_list[j - 1] != new_indices[cur - 1]):
                    counter[(new_indices[cur - 1], new_indices[cur])] += 1
                    pos_index[(new_indices[cur - 1], new_indices[cur])].add(pos)

                if cur + 1 < len(inner_pos_list):
                    counter[(new_indices[cur], new_indices[cur + 1])] += 1
                    pos_index[(new_indices[cur], new_indices[cur + 1])].add(pos)

        counter.pop(pair)
        pos_index.pop(pair)

    def _read_file_to_chunks(self, input_path: Union[str, os.PathLike]) -> list[str]:
        chunks = []

        with open(input_path, mode='r', encoding=BPEConstant.ENCODING) as file:
            while (text := file.read(BPEConstant.CHUNK_SIZE)) != '':
                chunks.append(text)

        return chunks

    def _split_by_patterns(self, text: str) -> list[bytes]:
        bytes_list = []
        parts = re.split(self.specials_pattern, text)

        for part in parts:
            tokens = regex.findall(BPEConstant.PRETOK_PATTERN, part)
            bytes_list += [token.encode(encoding=BPEConstant.ENCODING) for token in tokens]

        return bytes_list

    def fit(self, text: str):
        results = self._split_by_patterns(text)
        tokens: list[list[int]] = [list(token) for token in results]
        self.do_fit(tokens)

    @time_cost
    def do_fit(self, tokens: list[list[int]]):
        counter: defaultdict[tuple[int, int], int] = defaultdict(int)
        pos_index: defaultdict[tuple[int, int], set[int]] = defaultdict(set)

        for i, token in enumerate(tokens):
            for pair in zip(token, token[1:]):
                counter[pair] += 1
                pos_index[pair].add(i)

        num_merge = max(0, self.vocab_size - len(self.special_tokens) - BPEConstant.BASE_VOCAB_SIZE)
        for i in range(num_merge):
            max_count_pair = max(counter.items(), key=lambda x: (x[1], x[0][0], x[0][1]))[0]
            left, right = max_count_pair
            new_index = len(self.index2bytes)

            self.merges[max_count_pair] = new_index
            new_bytes = self.index2bytes[left] + self.index2bytes[right]
            self.index2bytes[new_index] = new_bytes
            self.bytes2index[new_bytes] = new_index

            self._merge_pair(counter, pos_index, tokens, max_count_pair, new_index)

    def _merge(self, indices: list[int], pair: tuple[int, int],
               new_index: int, counter: Counter = None,
               ) -> list[int]:
        left, right = pair
        i = 0
        n = len(indices)

        new_indices = [0] * n
        idx = 0

        if counter:
            counter.pop(pair)

        while i < n:
            if i < n - 1 and indices[i] == left and indices[i + 1] == right:
                new_indices[idx] = new_index
                idx += 1

                if counter:
                    if i > 0:
                        counter[(indices[i - 1], left)] -= 1
                        counter[(indices[i - 1], new_index)] += 1

                    if i < n - 2:
                        counter[(right, indices[i + 2])] -= 1
                        counter[(new_index, indices[i + 2])] += 1

                i += 2
            else:
                new_indices[idx] = indices[i]
                idx += 1
                i += 1

        return new_indices[:idx]


if __name__ == '__main__':
    a = 'hello world, nice to meet you,<|end|> how are you'
    tokenizer = BPETokenizer(5, ['<|end|>'])
    tokenizer.fit(a)

    indices = tokenizer.encode("worldto <|end|>you")
    revert = tokenizer.decode(indices)

    print('origin: ' + "worldto <|end|>you")
    print("encode: " + str(indices))
    print("decode: " + revert)
