# -- coding: utf-8 --
import os
import re
import time
from collections import defaultdict, Counter
from multiprocessing import Pool
from typing import Union
from loguru import logger

from models.tokenizer.consts import BPEConstant
from models.utils.linkedlist import *


class PairPosManager:
    def __init__(self, indices: list[int]):
        self.pos_dict: dict[tuple[int, int], list[Node]] = defaultdict(list)
        self.index_list: LinkedList = LinkedList(indices)

        cur = self.index_list.head
        while cur:
            if cur.next:
                self.pos_dict[(cur.data, cur.next.data)].append(cur)
            cur = cur.next

    def merge_pair(self, new_index: int, pair: tuple[int, int], counter: Counter):
        start_nodes = self.pos_dict[pair]
        if pair in counter:
            counter.pop(pair)

        for node in start_nodes:
            new_node = Node(new_index, node.prev, node.next.next)

            if node.prev:
                prev_pair = (node.prev.data, node.data)
                new_pair = (node.prev.data, new_index)
                counter[prev_pair] -= 1
                counter[new_pair] += 1

                self.pos_dict[prev_pair].remove(node.prev)
                self.pos_dict[new_pair].append(node.prev)

                node.prev.next = new_node
            else:
                self.index_list.head = new_node

            if node.next.next:
                next_pair = (node.next.data, node.next.next.data)
                new_pair = (new_index, node.next.next.data)
                counter[next_pair] -= 1
                counter[new_pair] += 1

                self.pos_dict[next_pair].remove(node.next)
                self.pos_dict[new_pair].append(new_node)

                node.next.next.prev = new_node
            else:
                self.index_list.tail = new_node


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

    def fit_file(self, input_path: Union[str, os.PathLike]):
        with open(input_path, mode='r', encoding=BPEConstant.ENCODING) as file:
            start = time.time()
            text = file.read()
            logger.info('read file cost: {}', time.time() - start)

            self.fit(text)

    def fit(self, text: str):
        start = time.time()
        num_merge = max(0, self.vocab_size - len(self.special_tokens) - BPEConstant.BASE_VOCAB_SIZE)
        logger.info('num merge: {}', num_merge)
        cur_vocab_size = len(self.index2bytes)
        split_text = self._get_split_text(text)
        split_indices = [list(map(int, content.encode(encoding=BPEConstant.ENCODING))) for content in split_text]
        logger.info('split indices cost: {}', time.time() - start)

        count_time = 0.0
        merge_time = 0.0
        start = time.time()
        counter = Counter()
        pair_managers: list[PairPosManager] = []

        count_start = time.time()
        for indices in split_indices:
            counter.update(zip(indices, indices[1:]))
            pair_managers.append(PairPosManager(indices))
        count_time += time.time() - count_start

        for i in range(num_merge):
            max_count_pair = max(counter.items(), key=lambda x: (x[1], x[0][0], x[0][1]))[0]
            left, right = max_count_pair
            new_index = cur_vocab_size + i

            self.merges[max_count_pair] = new_index
            new_bytes = self.index2bytes[left] + self.index2bytes[right]
            self.index2bytes[new_index] = new_bytes
            self.bytes2index[new_bytes] = new_index

            start_2 = time.time()
            for manager in pair_managers:
                manager.merge_pair(new_index, max_count_pair, counter)
            merge_time += time.time() - start_2

        logger.info('train cost: {}', time.time() - start)
        logger.info('count total cost: {}', count_time)
        logger.info('merge total cost: {}', merge_time)

    def _get_split_text(self, text: str) -> list[str]:
        if not self.special_tokens:
            return [text]
        else:
            return re.split(self.specials_pattern, text)

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
