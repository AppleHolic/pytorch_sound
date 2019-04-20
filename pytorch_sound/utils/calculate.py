import numpy as np
import torch
from pytorch_sound import settings as CFG


def db2log(db):
    if isinstance(db, np.ndarray) or isinstance(db, int):
        return np.log(np.power(10, db / 10))
    else:
        return torch.log(torch.pow(10, db / 10.))


def unnorm_mel(x: np.array) -> np.array:
    # mel range
    mel_min, mel_max = db2log(CFG.MIN_DB), db2log(CFG.MAX_DB)
    return ((x + 1) * 0.5) * (mel_max - mel_min) + mel_min


def norm_mel(x: np.array) -> np.array:
    # mel range
    mel_min, mel_max = db2log(CFG.MIN_DB), db2log(CFG.MAX_DB)
    return (x - mel_min) / (mel_max - mel_min) * 2 - 1
