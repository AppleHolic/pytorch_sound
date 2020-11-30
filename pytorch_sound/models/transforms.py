import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio import functional as audio_func
import librosa
import numpy as np
from scipy.signal import get_window, kaiser
from librosa.util import pad_center
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
        ) * fft_window

        self.register_buffer('square_window', fft_window ** 2)
        self.register_buffer('forward_basis', forward_basis)
        self.register_buffer('inverse_basis', inverse_basis)

    def transform(self, wav: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def inverse(self, magnitude: torch.Tensor, phase: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
        conc = torch.cat(
            [magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1)
        inverse_transform = F.conv_transpose1d(
            conc,
            self.inverse_basis,
            stride=self.hop_length,
            padding=0)

        # remove window effect
        n_frames = conc.size(-1)
        inverse_size = inverse_transform.size(-1)

        window_filter = torch.ones(
            1, 1, n_frames
        ).type_as(inverse_transform)

        weight = self.square_window[:self.filter_length].unsqueeze(0).unsqueeze(0)
        window_filter = F.conv_transpose1d(
            window_filter,
            weight,
            stride=self.hop_length,
            padding=0
        )
        indices = torch.arange(inverse_size)
        window_filter = window_filter.squeeze() + eps
        window_filter = window_filter[indices]

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
        self.register_buffer('mel_filter', torch.FloatTensor(mel_filter))

    def forward(self, wav: torch.Tensor, log_offset: float = 1e-6) -> torch.Tensor:
        mag, phase = self.stft.transform(wav)

        # apply mel filter
        mel = torch.matmul(self.mel_filter, mag)

        # to log-space
        mel = torch.log(mel + log_offset)

        return mel.clamp(self.min_db, self.max_db)


class LogMelScale(nn.Module):

    def __init__(self, sample_rate: int, mel_size: int, n_fft: int, min_db: float, max_db: float,
                 mel_min: float = 0., mel_max: float = None):
        super().__init__()
        self.mel_size = mel_size

        self.min_db = np.log(np.power(10, min_db / 10))
        self.max_db = np.log(np.power(10, max_db / 10))
        # mel filter banks
        mel_filter = librosa.filters.mel(sample_rate, n_fft, mel_size, fmin=mel_min, fmax=mel_max)
        self.register_buffer('mel_filter',
                             torch.Tensor(mel_filter, dtype=torch.float))

    def forward(self, magnitude: torch.Tensor, log_offset: float = 1e-6) -> torch.Tensor:
        # apply mel filter
        mel = torch.matmul(self.mel_filter, magnitude)

        # to log-space
        mel = torch.log(mel + log_offset)

        return mel.clamp(self.min_db, self.max_db)


class STFTTorchAudio(nn.Module):
    """
    Match interface between original one and pytorch official implementation
    """

    def __init__(self, filter_length: int = 1024, hop_length: int = 512, win_length: int = None, n_fft: int = None,
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
        if n_fft:
            self.n_fft = n_fft
        else:
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
        return torch.istft(
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

    def forward(self, wav: torch.Tensor, log_offset: float = 1e-6) -> torch.Tensor:
        # apply mel spectrogram
        mel = self.melfunc(wav)

        # to log-space
        mel = torch.log(mel + log_offset)

        return mel.clamp(self.min_db, self.max_db)


class SpectrogramMasker(nn.Module):
    """
    Helper class transforming wave-level mask to spectrogram-level mask
    """

    def __init__(self, win_length: int, hop_length: int):
        super().__init__()
        self.win_length = win_length
        self.conv = nn.Conv1d(
            1, 1, self.win_length, stride=hop_length, padding=0, bias=False).cuda()
        torch.nn.init.constant_(self.conv.weight, 1. / self.win_length)

    def forward(self, wav_mask: torch.Tensor) -> torch.Tensor:
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
        self.register_buffer('dct_mat', dct_mat.transpose(0, 1))

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
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
        self.register_buffer('dct_mat', dct_mat.transpose(0, 1))

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        assert len(wav.size()) == 3
        mel_spectrogram = self.mel_func(wav)
        return torch.matmul(self.dct_mat, mel_spectrogram)


#
# Pseudo QMF Module -
# Reference Code : https://github.com/kan-bayashi/ParallelWaveGAN/blob/master/parallel_wavegan/layers/pqmf.py
#
def design_prototype_filter(taps=62, cutoff_ratio=0.15, beta=9.0):
    """Design prototype filter for PQMF.
    This method is based on `A Kaiser window approach for the design of prototype
    filters of cosine modulated filterbanks`_.
    Args:
        taps (int): The number of filter taps.
        cutoff_ratio (float): Cut-off frequency ratio.
        beta (float): Beta coefficient for kaiser window.
    Returns:
        ndarray: Impluse response of prototype filter (taps + 1,).
    .. _`A Kaiser window approach for the design of prototype filters of cosine modulated filterbanks`:
        https://ieeexplore.ieee.org/abstract/document/681427
    """
    # check the arguments are valid
    assert taps % 2 == 0, "The number of taps mush be even number."
    assert 0.0 < cutoff_ratio < 1.0, "Cutoff ratio must be > 0.0 and < 1.0."

    # make initial filter
    omega_c = np.pi * cutoff_ratio
    h_i = np.sin(omega_c * (np.arange(taps + 1) - 0.5 * taps)) \
        / (np.pi * (np.arange(taps + 1) - 0.5 * taps))
    h_i[taps // 2] = np.cos(0) * cutoff_ratio  # fix nan due to indeterminate form

    # apply kaiser window
    w = kaiser(taps + 1, beta)
    h = h_i * w

    return h


class PQMF(torch.nn.Module):
    """PQMF module.
    This module is based on `Near-perfect-reconstruction pseudo-QMF banks`_.
    .. _`Near-perfect-reconstruction pseudo-QMF banks`:
        https://ieeexplore.ieee.org/document/258122
    """

    def __init__(self, subbands=4, taps=62, cutoff_ratio=0.15, beta=9.0):
        """Initilize PQMF module.
        Args:
            subbands (int): The number of subbands.
            taps (int): The number of filter taps.
            cutoff_ratio (float): Cut-off frequency ratio.
            beta (float): Beta coefficient for kaiser window.
        """
        super(PQMF, self).__init__()

        # define filter coefficient
        h_proto = design_prototype_filter(taps, cutoff_ratio, beta)
        h_analysis = np.zeros((subbands, len(h_proto)))
        h_synthesis = np.zeros((subbands, len(h_proto)))
        for k in range(subbands):
            h_analysis[k] = 2 * h_proto * np.cos(
                (2 * k + 1) * (np.pi / (2 * subbands)) *
                (np.arange(taps + 1) - ((taps - 1) / 2)) +
                (-1) ** k * np.pi / 4)
            h_synthesis[k] = 2 * h_proto * np.cos(
                (2 * k + 1) * (np.pi / (2 * subbands)) *
                (np.arange(taps + 1) - ((taps - 1) / 2)) -
                (-1) ** k * np.pi / 4)

        # convert to tensor
        h_analysis = torch.from_numpy(h_analysis).float().unsqueeze(1)
        h_synthesis = torch.from_numpy(h_synthesis).float().unsqueeze(0)

        # register coefficients as beffer
        self.register_buffer("analysis_filter", h_analysis)
        self.register_buffer("synthesis_filter", h_synthesis)

        # filter for downsampling & upsampling
        updown_filter = torch.zeros((subbands, subbands, subbands)).float()
        for k in range(subbands):
            updown_filter[k, k, 0] = 1.0
        self.register_buffer("updown_filter", updown_filter)
        self.subbands = subbands

        # keep padding info
        self.pad_fn = torch.nn.ConstantPad1d(taps // 2, 0.0)

    def analysis(self, x):
        """Analysis with PQMF.
        Args:
            x (Tensor): Input tensor (B, 1, T).
        Returns:
            Tensor: Output tensor (B, subbands, T // subbands).
        """
        x = F.conv1d(self.pad_fn(x), self.analysis_filter)
        return F.conv1d(x, self.updown_filter, stride=self.subbands)

    def synthesis(self, x):
        """Synthesis with PQMF.
        Args:
            x (Tensor): Input tensor (B, subbands, T // subbands).
        Returns:
            Tensor: Output tensor (B, 1, T).
        """
        # return F.conv1d(self.pad_fn(x), self.synthesis_filter)
        x = F.conv_transpose1d(x, self.updown_filter * self.subbands, stride=self.subbands)
        return F.conv1d(self.pad_fn(x), self.synthesis_filter)
