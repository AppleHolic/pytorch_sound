import torch.nn as nn
import librosa
import torch
from torch_stft import stft


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

    def forward(self, wav, eps=1e-7):
        mag, phase = self.stft.transform(wav)

        # apply mel filter
        mel = torch.matmul(self.mel_filter, mag)

        # clip
        mel = mel.clamp(self.min_db, self.max_db)

        # to log-space
        mel = torch.log(mel)

        return mel
