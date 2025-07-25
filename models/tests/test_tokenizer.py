from __future__ import annotations

import json
import os
import sys

import psutil
import pytest
import tiktoken

from models.tests.adapters import get_tokenizer
from models.tests.common import FIXTURES_PATH, gpt2_bytes_to_unicode

VOCAB_PATH = FIXTURES_PATH / "gpt2_vocab.json"
MERGES_PATH = FIXTURES_PATH / "gpt2_merges.txt"


def memory_limit(max_mem):
    """跨平台内存限制装饰器"""

    def decorator(f):
        def wrapper(*args, **kwargs):
            if sys.platform.startswith('linux'):
                # Linux 平台使用 resource 模块
                import resource
                process = psutil.Process(os.getpid())
                prev_limits = resource.getrlimit(resource.RLIMIT_AS)
                resource.setrlimit(resource.RLIMIT_AS, (process.memory_info().rss + max_mem, -1))
                try:
                    result = f(*args, **kwargs)
                    return result
                finally:
                    resource.setrlimit(resource.RLIMIT_AS, prev_limits)
            else:
                # 非Linux平台直接执行函数
                return f(*args, **kwargs)

        return wrapper

    return decorator


def get_tokenizer_from_vocab_merges_path(
        vocab_path: str | os.PathLike,
        merges_path: str | os.PathLike,
        special_tokens: list[str] | None = None,
):
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(vocab_path) as vocab_f:
        gpt2_vocab = json.load(vocab_f)
    gpt2_bpe_merges = []
    with open(merges_path) as f:
        for line in f:
            cleaned_line = line.rstrip()
            if cleaned_line and len(cleaned_line.split(" ")) == 2:
                gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))

    vocab = {
        gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
        for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
    }

    if special_tokens:
        for special_token in special_tokens:
            byte_encoded_special_token = special_token.encode("utf-8")
            if byte_encoded_special_token not in set(vocab.values()):
                vocab[len(vocab)] = byte_encoded_special_token

    merges = [
        (
            bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
            bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
        )
        for merge_token_1, merge_token_2 in gpt2_bpe_merges
    ]
    return get_tokenizer(vocab, merges, special_tokens)


# 保留所有测试函数不变...
# 只需要修改与内存限制相关的部分

@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="内存限制功能仅在Linux平台完全支持",
)
def test_encode_iterable_memory_usage():
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    with open(FIXTURES_PATH / "tinystories_sample_5M.txt") as f:
        ids = []
        for _id in _encode_iterable(tokenizer, f):
            ids.append(_id)


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="内存限制功能仅在Linux平台完全支持",
)
@pytest.mark.xfail(reason="Tokenizer.encode 预期会使用超过分配的内存(1MB)")
def test_encode_memory_usage():
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )
    with open(FIXTURES_PATH / "tinystories_sample_5M.txt") as f:
        contents = f.read()
        _ = _encode(tokenizer, contents)


@memory_limit(int(1e6))
def _encode_iterable(tokenizer, iterable):
    yield from tokenizer.encode_iterable(iterable)


@memory_limit(int(1e6))
def _encode(tokenizer, text):
    return tokenizer.encode(text)