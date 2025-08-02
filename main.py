import time

import torch.cuda
from loguru import logger

from models.tokenizer.bpe import BPETokenizer
from models.tokenizer.consts import BPEConstant
from models.transformer.loss import cross_entropy_loss
from models.transformer.optimizers import AdamW, cos_lr_schedule
from models.transformer.transformer import TransformerLM
from models.utils.data_utils import transfer_dataset, save_bpe_tokenizer, load_bpe_tokenizer
from models.utils.file_utils import read_file_to_chunks
from models.utils.train_utils import train_llm


if __name__ == '__main__':
    # with open('./models/data/TinyStoriesV2-GPT4-train.txt', mode='r', encoding=BPEConstant.ENCODING) as f:
    #     with open('./models/data/train-text-20M.txt', mode='w+', encoding=BPEConstant.ENCODING) as out:
    #         out.write(f.read(100 * 1024 * 1024))

    # tokenizer = BPETokenizer(vocab_size=10000, special_tokens=['<|endoftext|>'])
    # logger.info('start train bpe')
    # tokenizer.fit_file('./models/data/train-text-20M.txt')
    # logger.info('bpe training completed')

    import faulthandler

    faulthandler.enable()
    tokenizer = load_bpe_tokenizer('./models/data/bpe.pickle')

    logger.info('start transfer data')
    train_chunks = read_file_to_chunks('./models/data/TinyStoriesV2-GPT4-train.txt')
    test_chunks = read_file_to_chunks('./models/data/TinyStoriesV2-GPT4-valid.txt')
    logger.info('complete data transfer')

    device = 'cpu'
    logger.info(f'device: {device}')
    model_params = {
        'd_model': 512,
        'num_heads': 16,
        'd_ff': 1344,
        'max_seq_len': 256,
        'vocab_size': 10000,
        'theta': 10000,
        'num_layers': 4,
        'device': device
    }
    optim_params = {
    }

    scheduler_params = {
        'lr_max': 1,
        'lr_min': 0.001,
        't_w': 20,
        't_c': 50,
    }

    train_llm(tokenizer, train_chunks, test_chunks, TransformerLM, model_params,
              AdamW, optim_params, cross_entropy_loss, cos_lr_schedule, scheduler_params,
              max_l2_norm=5, batch_size=128, epochs=10000, context_len=256, device=device)
