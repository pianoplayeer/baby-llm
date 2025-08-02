# -- coding: utf-8 --
import re
from typing import Final

from regex import regex


class BPEConstant:
    BASE_VOCAB_SIZE: Final[int] = 256
    ENCODING = 'utf-8'
    CHUNK_SIZE = 4096
    PRETOK_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    BPE_PATH = './bpe.pickle'


if __name__ == '__main__':
    a = 'hello, my name is jxy. nice to, meet  !'
    l = re.findall(BPEConstant.PRETOK_PATTERN, a)
    print(l)