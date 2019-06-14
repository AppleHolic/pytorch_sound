import numpy as np
import torch
from pytorch_sound import settings


def db2log(db):
    if isinstance(db, np.ndarray) or isinstance(db, int):
        return np.log(np.power(10, db / 10))
    else:
        return torch.log(torch.pow(10, db / 10.))


def unnorm_mel(x):
    # mel range
    mel_min, mel_max = db2log(settings.MIN_DB), db2log(settings.MAX_DB)
    return ((x + 1) * 0.5) * (mel_max - mel_min) + mel_min


def norm_mel(x):
    # mel range
    mel_min, mel_max = db2log(settings.MIN_DB), db2log(settings.MAX_DB)
    return (x - mel_min) / (mel_max - mel_min) * 2 - 1


def volume_norm_log(x: np.array, target_db: float = -11.5) -> np.array:
    return x / (np.std(x) / 10 ** (target_db / 10))


def volume_norm_log_torch(x: torch.tensor, target_db: float = -11.5) -> torch.tensor:
    return x / (torch.std(x) / 10 ** (target_db / 10))
