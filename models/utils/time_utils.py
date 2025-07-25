# -- coding: utf-8 --
import time
from functools import wraps

from loguru import logger


def time_cost(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration = time.perf_counter() - start
        logger.info(f'{func} time cost: {duration}')
        return result
    return wrapper


@time_cost
def count():
    a = 1
    for i in range(10000000):
        a += i

if __name__ == '__main__':
    count()