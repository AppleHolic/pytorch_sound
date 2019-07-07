import torch.nn as nn
import librosa
import torch
from torch_stft import stft
try:
    from torchaudio.transforms import MelSpectrogram as MelJit
except ImportError:
    MelJit = None


class MelSpectrogram(nn.Module):

    def __init__(self, sample_rate: int, mel_size: int, n_fft: int, win_length: int,
                 hop_length: int, min_db: float, max_db: float,
                 mel_min: float = 0., mel_max: float = None):
        super().__init__()
        self.mel_size = mel_size
        self.min_db = min_db
        self.max_db = max_db

        self.stft = stft.STFT(filter_length=win_length, hop_length=hop_length)

        # mel filter banks
        mel_filter = librosa.filters.mel(sample_rate, n_fft, mel_size, fmin=mel_min, fmax=mel_max)
        self.register_buffer('mel_filter',
                             torch.tensor(mel_filter, dtype=torch.float))

    def forward(self, wav: torch.tensor, log_offset: float = 1e-6) -> torch.tensor:
        mag, phase = self.stft.transform(wav)

        # apply mel filter
        mel = torch.matmul(self.mel_filter, mag)

        # clip
        mel = mel.clamp(self.min_db, self.max_db)

        # to log-space
        mel = torch.log(mel + log_offset)

        return mel


class MelMasker(nn.Module):

    def __init__(self, win_length: int, hop_length: int):
        super().__init__()
        self.conv = nn.Conv1d(
            1, 1, win_length, stride=hop_length, padding=win_length // 2, bias=False).cuda()
        torch.nn.init.constant_(self.conv.weight, 1.)

    def forward(self, wav_mask):
        # make mask
        with torch.no_grad():
            mel_mask = self.conv(wav_mask.float().unsqueeze(1)).squeeze(1)
            mel_mask = (mel_mask > 0).float()
        return mel_mask


class MelToMFCC(nn.Module):

    def __init__(self, n_mfcc: int, mel_size: int):
        super().__init__()
        self.n_mfcc = n_mfcc
        # register mfcc dct filter
        self.register_buffer('mfcc_filter',
                             torch.FloatTensor(librosa.filters.dct(self.n_mfcc, mel_size)).unsqueeze(0))

    def forward(self, mel_spec: torch.tensor) -> torch.tensor:
        assert len(mel_spec.size()) == 3
        return torch.matmul(self.mfcc_filter, mel_spec)


class MFCC(nn.Module):

    def __init__(self, sample_rate: int, mel_size: int, n_fft: int, win_length: int, n_mfcc: int,
                 hop_length: int, min_db: float, max_db: float,
                 mel_min: float = 0., mel_max: float = None):
        super().__init__()
        self.n_mfcc = n_mfcc
        self.mel_func = MelSpectrogram(
            sample_rate, mel_size, n_fft, win_length, hop_length, min_db, max_db,
            mel_min, mel_max
        )
        # register mfcc dct filter
        self.register_buffer('mfcc_filter',
                             torch.FloatTensor(librosa.filters.dct(self.n_mfcc, mel_size)).unsqueeze(0))

    def forward(self, wav: torch.tensor) -> torch.tensor:
        assert len(wav.size()) == 3
        mel_spectrogram = self.mel_func(wav)
        return torch.matmul(self.mfcc_filter, mel_spectrogram)


#
# torchaudio jit computation version.
#
class MelSpectrogramJIT(nn.Module):

    def __init__(self, sample_rate: int, mel_size: int, n_fft: int, win_length: int,
                 hop_length: int, min_db: float, max_db: float,
                 mel_min: float = 0., mel_max: float = None):
        super().__init__()
        if MelJit is None:
            raise NotImplementedError('You should install torchaudio to use it!')

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
