import torch.nn as nn
import librosa
import torch
from torch_stft import stft
from torchaudio.transforms import MelSpectrogram as MelJit


class MelSpectrogram(nn.Module):

    def __init__(self, sample_rate: int, mel_size: int, n_fft: int, win_length: int,
                 hop_length: int, min_db: float, max_db: float,
                 mel_min: float = 0., mel_max: float = None):
        super().__init__()
        self.min_db = min_db
        self.max_db = max_db

        self.stft = stft.STFT(filter_length=win_length, hop_length=hop_length)

        # mel filter banks
        mel_filter = librosa.filters.mel(sample_rate, n_fft, mel_size, fmin=mel_min, fmax=mel_max)
        self.register_buffer('mel_filter',
                             torch.tensor(mel_filter, dtype=torch.float))

    def forward(self, wav: torch.tensor) -> torch.tensor:
        mag, phase = self.stft.transform(wav)

        # apply mel filter
        mel = torch.matmul(self.mel_filter, mag)

        # clip
        mel = mel.clamp(self.min_db, self.max_db)

        # to log-space
        mel = torch.log(mel)

        return mel


#
# torchaudio jit computation version.
#
class MelSpectrogramJIT(nn.Module):

    def __init__(self, sample_rate: int, mel_size: int, n_fft: int, win_length: int,
                 hop_length: int, min_db: float, max_db: float,
                 mel_min: float = 0., mel_max: float = None):
        super().__init__()
        self.mel_func = MelJit(sr=sample_rate, n_fft=n_fft, ws=win_length, hop=hop_length, f_min=float(mel_min),
                               f_max=float(mel_max), pad=win_length // 2, n_mels=mel_size, window=torch.hann_window,
                               wkwargs=None)
        self.min_db = min_db
        self.max_db = max_db

    def forward(self, wav):
        # make mel
        melspec = self.mel_func(wav).transpose(1, 2)

        # clamp
        melspec = melspec.clamp(self.min_db, self.max_db)

        return torch.log(melspec)
