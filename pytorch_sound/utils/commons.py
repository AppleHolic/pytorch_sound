import multiprocessing
import logging
import torch
from typing import Dict, Callable, List, Any


__all__ = ['LOGGER', 'go_multiprocess']


def go_multiprocess(worker_func: Callable, inputs: List[Any]) -> List[Any]:
    """
    Run and return worker function using multiprocessing.Pool.
    :param worker_func: callable worker function
    :param inputs: list of arguments for worker function
    :return: results
    """

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
    """
    Get formatted logger instance
    :param name: logger's name
    :return: instance of logger
    """
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
    """
    print message with using global logger instance
    :param msg: message to be printed
    """
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
