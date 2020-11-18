import logging
import torch
from tqdm import tqdm
from joblib import Parallel, delayed, cpu_count
from typing import Dict, Callable, List, Any, Tuple

__all__ = ['LOGGER', 'go_multiprocess']


def go_multiprocess(worker_func: Callable, inputs: List[Tuple[Any]], num_workers: int = None) -> List[Any]:
    """
    Run and return worker function using joblib.
    :param worker_func: callable worker function
    :param inputs: list of arguments for worker function
    :return: results
    """
    # TODO: update code lines using this function
    if not num_workers:
        num_workers = cpu_count() // 2

    results = Parallel(n_jobs=num_workers)(delayed(worker_func)(*args) for args in tqdm(inputs))
    return results


def get_logger(name: str):
    """
    Get formatted logger instance
    :param name: logger's name
    :return: instance of logger
    """
    # setup logger
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.propagate = False
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
