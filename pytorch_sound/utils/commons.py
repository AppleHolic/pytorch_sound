import multiprocessing
import logging
import torch
from typing import Dict, Callable, List, Any


__all__ = ['LOGGER']


def go_multiprocess(worker_func: Callable, inputs: List[Any]) -> List[Any]:

    # declare pool
    cpu_count = multiprocessing.cpu_count() // 2

    res = []

    with multiprocessing.Pool(cpu_count) as pool:
        for i in range(0, len(inputs), cpu_count):
            start_idx, end_idx = i, i + cpu_count
            res += pool.map(worker_func, inputs[start_idx:end_idx])
            print('{}/{}\t{}() processed.'.format(i + 1, len(inputs), worker_func.__name__))

    return res


def get_logger(name: str):
    # setup logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


LOGGER = get_logger('main')


def log(msg: str):
    LOGGER.info(msg)


def get_loadable_checkpoint(checkpoint: Dict[str, torch.tensor]) -> Dict[str, torch.tensor]:
    """
    If model is saved with DataParallel, checkpoint keys is started with 'module.' remove it and return new state dict
    :param checkpoint:
    :return: new checkpoint
    """
    new_checkpoint = {}
    for key, val in checkpoint.items():
        new_key = key.replace('module.', '')
        new_checkpoint[new_key] = val
    return new_checkpoint
