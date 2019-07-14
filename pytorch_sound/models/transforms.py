import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
from scipy.signal import get_window
from librosa.util import pad_center
try:
    from torchaudio.transforms import MelSpectrogram as MelJit
except ImportError:
    MelJit = None


#
# Re-construct stft for calculating backward operation
# refer on : https://github.com/pseeth/torch-stft/blob/master/torch_stft/stft.py
#
class STFT(nn.Module):

    def __init__(self, filter_length=1024, hop_length=512, win_length=None,
                 window='hann'):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length if win_length else filter_length
        self.window = window
        self.forward_transform = None
        self.pad_amount = self.filter_length // 2

        # make fft window
        assert (filter_length >= self.win_length)
        # get window and zero center pad it to filter_length
        fft_window = get_window(window, self.win_length, fftbins=True)
        fft_window = pad_center(fft_window, filter_length)
        fft_window = torch.from_numpy(fft_window).float()

        # calculate fourer_basis
        cut_off = int((self.filter_length / 2 + 1))
        fourier_basis = np.fft.fft(np.eye(self.filter_length))
        fourier_basis = np.vstack([
            np.real(fourier_basis[:cut_off, :]),
            np.imag(fourier_basis[:cut_off, :])
        ])

        # make forward & inverse basis
        forward_basis = torch.FloatTensor(fourier_basis[:, np.newaxis, :]) * fft_window
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(self.filter_length / self.hop_length * fourier_basis).T[:, np.newaxis, :]
        )

        self.register_buffer('square_window', fft_window ** 2)
        self.register_buffer('forward_basis', forward_basis)
        self.register_buffer('inverse_basis', inverse_basis)

    def transform(self, wav):
        # reflect padding
        wav = wav.unsqueeze(1).unsqueeze(1)
        wav = F.pad(
            wav,
            (self.pad_amount, self.pad_amount, 0, 0),
            mode='reflect'
        ).squeeze(1)

        # conv
        forward_trans = F.conv1d(
            wav, self.forward_basis,
            stride=self.hop_length, padding=0
        )
        real_part, imag_part = forward_trans.chunk(2, 1)

        return torch.sqrt(real_part ** 2 + imag_part ** 2), torch.atan2(imag_part.data, real_part.data)

    def inverse(self, magnitude, phase, eps=1e-9):
        conc = torch.cat(
            [magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1)

        inverse_transform = F.conv_transpose1d(
            conc,
            self.inverse_basis,
            stride=self.hop_length,
            padding=0)

        # remove window effect
        if self.window is not None:
            n_frames = conc.size(-1)
            inverse_size = inverse_transform.size(-1)
            window_filter = torch.zeros(
                inverse_size
            ).type_as(inverse_transform).fill_(eps)

            for idx in range(n_frames):
                sample = idx * self.hop_length
                window_filter[sample:min(inverse_size, sample + self.filter_length)] \
                    += self.square_window[:max(0, min(self.filter_length, inverse_size - sample))]

            inverse_transform /= window_filter

            # scale by hop ratio
            inverse_transform *= self.filter_length / self.hop_length

        return inverse_transform[..., self.pad_amount:-self.pad_amount].squeeze(1)


class MelSpectrogram(nn.Module):

    def __init__(self, sample_rate: int, mel_size: int, n_fft: int, win_length: int,
                 hop_length: int, min_db: float, max_db: float,
                 mel_min: float = 0., mel_max: float = None):
        super().__init__()
        self.mel_size = mel_size
        self.min_db = min_db
        self.max_db = max_db

        self.stft = STFT(filter_length=win_length, hop_length=hop_length)

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
