import numpy as np
import torch
from typing import Any


def to_device(tup: Any) -> torch.tensor:
    """
    helper function to make cuda tensor from a tuple or a list of tensors
    :param tup: a tuple or a list of tensors
    :return: cuda tensors
    """

    if not (isinstance(tup, tuple) or isinstance(tup, list)):
        tup = (tup,)
    return map(lambda x: x.cuda(non_blocking=True), tup)


def to_numpy(gpu_tensor: torch.tensor) -> np.ndarray:
    """
    Make numpy array from cuda tensor
    :param gpu_tensor: cuda tensor
    :return: numpy array
    """
    return gpu_tensor.detach().cpu().numpy()


def concat_complex(a: torch.tensor, b: torch.tensor, dim: int = 1) -> torch.tensor:
    """
    Concatenate two complex tensors in same dimension concept
    :param a: complex tensor
    :param b: another complex tensor
    :param dim: target dimension
    :return: concatenated tensor
    """
    a_real, a_img = a.chunk(2, dim)
    b_real, b_img = b.chunk(2, dim)
    return torch.cat([a_real, b_real, a_img, b_img], dim=dim)
