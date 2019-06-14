import numpy as np
import torch
from typing import Any


def to_device(tup: Any) -> torch.tensor:
    if not (isinstance(tup, tuple) or isinstance(tup, list)):
        tup = (tup,)
    return map(lambda x: x.cuda(non_blocking=True), tup)


def to_numpy(gpu_tensor: torch.tensor) -> np.array:
    return gpu_tensor.detach().cpu().numpy()


def concat_complex(a, b, dim=1):
    a_real, a_img = a.chunk(2, dim)
    b_real, b_img = b.chunk(2, dim)
    return torch.cat([a_real, b_real, a_img, b_img], dim=dim)
