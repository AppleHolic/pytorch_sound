import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio import functional as audio_func
import librosa
import numpy as np
from scipy.signal import get_window
from librosa.util import pad_center
from torchaudio.functional import istft
from torchaudio.transforms import MelSpectrogram
from typing import Tuple


class STFT(nn.Module):
    """
    Re-construct stft for calculating backward operation
    refer on : https://github.com/pseeth/torch-stft/blob/master/torch_stft/stft.py
    """

    def __init__(self, filter_length: int = 1024, hop_length: int = 512, win_length: int = None,
                 window: str = 'hann'):
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

    def transform(self, wav: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
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

    def inverse(self, magnitude: torch.tensor, phase: torch.tensor, eps: float = 1e-9) -> torch.tensor:
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


class LogMelSpectrogram(nn.Module):
    """
    Mel spectrogram module with above STFT class
    """

    def __init__(self, sample_rate: int, mel_size: int, n_fft: int, win_length: int,
                 hop_length: int, min_db: float, max_db: float,
                 mel_min: float = 0., mel_max: float = None):
        super().__init__()
        self.mel_size = mel_size

        self.min_db = np.log(np.power(10, min_db / 10))
        self.max_db = np.log(np.power(10, max_db / 10))

        self.stft = STFT(filter_length=win_length, hop_length=hop_length)

        # mel filter banks
        mel_filter = librosa.filters.mel(sample_rate, n_fft, mel_size, fmin=mel_min, fmax=mel_max)
        self.register_buffer('mel_filter',
                             torch.tensor(mel_filter, dtype=torch.float))

    def forward(self, wav: torch.tensor, log_offset: float = 1e-6) -> torch.tensor:
        mag, phase = self.stft.transform(wav)

        # apply mel filter
        mel = torch.matmul(self.mel_filter, mag)

        # to log-space
        mel = torch.log(mel + log_offset)

        return mel.clamp(self.min_db, self.max_db)


class STFTTorchAudio(nn.Module):
    """
    Match interface between original one and pytorch official implementation
    """

    def __init__(self, filter_length: int = 1024, hop_length: int = 512, win_length: int = None,
                 window: str = 'hann'):
        super().__init__()
        # original arguments
        self.filter_length = filter_length
        self.hop_length = hop_length
        if win_length:
            self.win_length = win_length
        else:
            self.win_length = self.filter_length
        if window == 'hann':
            self.register_buffer('window', torch.hann_window(self.win_length))
        else:
            raise NotImplemented(f'{window} is not implemented ! Use hann')

        # pytorch official arguments
        self.n_fft = self.win_length

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        stft = torch.stft(
            wav, self.n_fft, self.hop_length, self.win_length, self.window, True,
            'reflect', False, True
        )  # (N, C, T, 2)
        real_part, img_part = [x.squeeze(3) for x in stft.chunk(2, 3)]
        return real_part, img_part

    def transform(self, wav: torch.Tensor) -> torch.Tensor:
        """
        :param wav: wave tensor
        :return: (N, Spec Dimension * 2, T) 3 dimensional stft tensor
        """
        real_part, img_part = self.forward(wav)
        return torch.sqrt(real_part ** 2 + img_part ** 2), torch.atan2(img_part, real_part)

    def inverse(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        # match dimension
        magnitude, phase = magnitude.unsqueeze(3), phase.unsqueeze(3)
        stft = torch.cat([magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=3)
        return istft(
            stft, self.n_fft, self.hop_length, self.win_length, self.window
        )


class Audio2Mel(nn.Module):
    """
    MelGAN's Log Mel Spectrogram Module
    - refer: https://github.com/descriptinc/melgan-neurips/blob/master/mel2wav/modules.py
    """
    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        sampling_rate: int = 22050,
        n_mel_channels: int = 80,
        mel_fmin: int = 0.0,
        mel_fmax: int = None,
    ):
        super().__init__()
        window = torch.hann_window(win_length).float()
        mel_basis = librosa.filters.mel(
            sampling_rate, n_fft, n_mel_channels, mel_fmin, mel_fmax
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", window)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels

    def forward(self, audio):
        p = (self.n_fft - self.hop_length) // 2
        audio = F.pad(audio, (p, p), "reflect").squeeze(1)
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
        )
        real_part, imag_part = fft.unbind(-1)
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        mel_output = torch.matmul(self.mel_basis, magnitude)
        log_mel_spec = torch.log10(torch.clamp(mel_output, min=1e-5))
        return log_mel_spec


class LogMelSpectrogramTorchAudio(nn.Module):
    """
    Mel spectrogram module with above STFT class
    """

    def __init__(self, sample_rate: int, mel_size: int, n_fft: int, win_length: int,
                 hop_length: int, min_db: float, max_db: float,
                 mel_min: float = 0., mel_max: float = None):
        super().__init__()
        self.mel_size = mel_size
        # db to log
        self.min_db = np.log(np.power(10, min_db / 10))
        self.max_db = np.log(np.power(10, max_db / 10))

        self.melfunc = MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, win_length=win_length,
                                      hop_length=hop_length, f_min=mel_min, f_max=mel_max, n_mels=mel_size,
                                      window_fn=torch.hann_window)

    def forward(self, wav: torch.tensor, log_offset: float = 1e-6) -> torch.tensor:
        # apply mel spectrogram
        mel = self.melfunc(wav)

        # to log-space
        mel = torch.log(mel + log_offset)

        return mel.clamp(self.min_db, self.max_db)


class MelMasker(nn.Module):
    """
    Helper class transforming wave-level mask to spectrogram-level mask
    """

    def __init__(self, win_length: int, hop_length: int):
        super().__init__()
        self.win_length = win_length
        self.conv = nn.Conv1d(
            1, 1, self.win_length, stride=hop_length, padding=0, bias=False).cuda()
        torch.nn.init.constant_(self.conv.weight, 1. / self.win_length)

    def forward(self, wav_mask: torch.tensor) -> torch.tensor:
        # make mask
        with torch.no_grad():
            wav_mask = F.pad(wav_mask, [0, self.win_length // 2], value=0.)
            wav_mask = F.pad(wav_mask, [self.win_length // 2, 0], value=1.)
            mel_mask = self.conv(wav_mask.float().unsqueeze(1)).squeeze(1)
            mel_mask = torch.ceil(mel_mask)
        return mel_mask


class MelToMFCC(nn.Module):
    """
    Create the Mel-frequency cepstrum coefficients from mel-spectrogram
    """

    def __init__(self, n_mfcc: int, mel_size: int, norm: str = 'ortho'):
        super().__init__()
        self.n_mfcc = n_mfcc
        dct_mat = audio_func.create_dct(n_mfcc, mel_size, norm)
        self.register_buffer('dct_mat', dct_mat)

    def forward(self, mel_spec: torch.tensor) -> torch.tensor:
        assert len(mel_spec.size()) == 3
        return torch.matmul(self.dct_mat, mel_spec)


class MFCC(nn.Module):
    """
    Create the Mel-frequency cepstrum coefficients from an audio signal
    """

    def __init__(self, sample_rate: int, mel_size: int, n_fft: int, win_length: int, n_mfcc: int,
                 hop_length: int, min_db: float, max_db: float,
                 mel_min: float = 0., mel_max: float = None, norm: str = 'ortho'):
        super().__init__()
        self.n_mfcc = n_mfcc
        self.mel_func = LogMelSpectrogram(
            sample_rate, mel_size, n_fft, win_length, hop_length, min_db, max_db,
            mel_min, mel_max
        )
        dct_mat = audio_func.create_dct(n_mfcc, mel_size, norm)
        self.register_buffer('dct_mat', dct_mat)

    def forward(self, wav: torch.tensor) -> torch.tensor:
        assert len(wav.size()) == 3
        mel_spectrogram = self.mel_func(wav)
        return torch.matmul(self.dct_mat, mel_spectrogram)
