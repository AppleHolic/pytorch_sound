import numpy as np
import torch
from typing import Any


def to_device(tup: Any):
    if not (isinstance(tup, tuple) or isinstance(tup, list)):
        tup = (tup,)
    return map(lambda x: x.cuda(non_blocking=True), tup)


def to_numpy(gpu_tensor) -> np.array:
    return gpu_tensor.detach().cpu().numpy()


def fix_length(x: np.array, size: int, axis: int = -1) -> np.array:
    n = x.shape[axis]
    if n > size:
        slices = [slice(None)] * x.ndim
        slices[axis] = slice(0, size)
        return x[tuple(slices)]
    elif n < size:
        lengths = [(0, 0)] * x.ndim
        lengths[axis] = (0, size - n)
        return np.pad(x, lengths, mode='constant')
    else:
        return x


def concat_complex(a, b, dim=1):
    a_real, a_img = a.chunk(2, dim)
    b_real, b_img = b.chunk(2, dim)
    return torch.cat([a_real, b_real, a_img, b_img], dim=dim)
