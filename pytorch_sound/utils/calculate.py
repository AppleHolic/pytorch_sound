import numpy as np
import torch
from typing import Union
from pytorch_sound import settings


TensorOrArr = Union[torch.tensor, np.ndarray]


def db2log(db: TensorOrArr) -> TensorOrArr:
    """
    convert audio data from db scale to log scale
    :param db: db scale audio
    :return: log scale audio
    """
    if isinstance(db, np.ndarray) or isinstance(db, int):
        return np.log(np.power(10, db / 10))
    else:
        return torch.log(torch.pow(10, db / 10.))


def unnorm_mel(x: TensorOrArr) -> TensorOrArr:
    """
    reverse normalized mel
    :param x: normalized mel spectrogram
    :return: log mel spectrogram
    """
    mel_min, mel_max = db2log(settings.MIN_DB), db2log(settings.MAX_DB)
    return ((x + 1) / 2) * (mel_max - mel_min) + mel_min


def norm_mel(x: TensorOrArr) -> TensorOrArr:
    """
    normalize mel spectrogram from -1 to +1
    :param x: log mel spectrogram
    :return: normalized mel spectrogram
    """
    mel_min, mel_max = db2log(settings.MIN_DB), db2log(settings.MAX_DB)
    x = x.clamp(mel_min, mel_max)
    return (x - mel_min) / (mel_max - mel_min) * 2 - 1


def volume_norm_log(x: np.array, target_db: float = -11.5) -> np.array:
    """
    rms volume normalization with numpy
    :param x: wave data
    :param target_db: target decibel
    :return: normalized wave
    """
    return x / (np.std(x) / 10 ** (target_db / 10))


def volume_norm_log_torch(x: torch.tensor, target_db: float = -11.5) -> torch.tensor:
    """
    rms volume normalization with pytorch
    :param x: wave data
    :param target_db: target decibel
    :return: normalized wave
    """
    return x / (torch.std(x) / 10 ** (target_db / 10))


def conv_same_padding(filter_size: int, stride: int, dilation: int, x: int = 44100) -> int:
    """
    :return: "same" padding size using given arguments
    """
    return int(np.ceil(((x / stride - 1) * stride + (filter_size + (filter_size - 1) * (dilation - 1)) - x) / 2))
