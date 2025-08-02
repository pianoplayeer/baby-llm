# -- coding: utf-8 --
import os
from typing import Union

from models.tokenizer.consts import BPEConstant


def read_file_to_chunks(input_path: Union[str, os.PathLike]) -> list[str]:
    chunks = []

    with open(input_path, mode='r', encoding=BPEConstant.ENCODING) as file:
        while (text := file.read(BPEConstant.CHUNK_SIZE)) != '':
            chunks.append(text)

    return chunks

