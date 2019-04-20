import os
import librosa
import numpy as np
import torch
from librosa.util import pad_center
from scipy.signal import get_window


#
# based on :
# - https://github.com/pseeth/pytorch-stft/blob/master/stft.py
# - https://github.com/NVIDIA/tacotron2/blob/fc0cf6a89a47166350b65daa1beaa06979e4cddf/stft.py


def _get_window(filter_len, name='hann'):
    return get_window(name, filter_len, fftbins=True)


def _apply_window(x, filter_len):
    # apply window
    filter = _get_window(filter_len)
    filter = pad_center(filter, filter_len)
    return x * filter


# calc basis for STFT
def _get_forward_basis(filter_len, apply_window=True):
    # get basis
    half_length = int((filter_len / 2 + 1))
    basis = np.fft.fft(np.eye(filter_len))
    basis = np.vstack([np.real(basis[:half_length, :]),
                       np.imag(basis[:half_length, :])])
    # apply window
    if apply_window:
        return _apply_window(basis, filter_len)
    else:
        return basis


def _get_inverse_basis(filter_len, hop_len, cache_file='_istft_basis.npy'):

    if os.path.exists(cache_file):
        return np.load(cache_file)
    else:
        # get basis
        basis = _get_forward_basis(filter_len, apply_window=False)
        basis = np.linalg.pinv(basis * filter_len / hop_len).T
        # apply window
        basis = _apply_window(basis, filter_len)
        np.save(cache_file, basis)
        return basis


class STFT(torch.nn.Module):

    def __init__(self, filter_len, hop_len):
        super().__init__()
        self.filter_len = filter_len
        self.hop_len = hop_len
        # save basis as buffer
        self.register_buffer('basis',
                             torch.tensor(_get_forward_basis(self.filter_len),
                                          dtype=torch.float).unsqueeze(1))

    def forward(self, x):
        # similar to librosa, reflect-pad the input
        x = x.view(x.shape[0], 1, x.shape[1])
        x = torch.nn.functional.pad(
            x.unsqueeze(1), (self.filter_len // 2, self.filter_len // 2, 0, 0),
            mode='reflect').squeeze(1)

        # do STFT by 1D convolution
        return torch.conv1d(
            x, self.basis, stride=self.hop_len, padding=0)

    @staticmethod
    def to_mag_phase(stft):
        # split real and imaginary parts
        re, im = torch.split(stft, stft.shape[1] // 2, dim=1)
        # calc magnitude and phase
        return torch.sqrt(re ** 2 + im ** 2), torch.atan2(im, re)

    @staticmethod
    def to_stft(mag, phase):
        # recombine magnitude and phase
        return torch.cat([mag * torch.cos(phase), mag * torch.sin(phase)], dim=1)


class MelSpectrogram(torch.nn.Module):

    def __init__(self, sample_rate, n_fft, hop_len, mel_size, mel_min, mel_fmax, min_db):
        super(MelSpectrogram, self).__init__()
        self.mel_size = mel_size
        self.min_db = min_db

        # mel filter banks
        mel_filters = librosa.filters.mel(sample_rate, n_fft, mel_size, fmin=mel_min, fmax=mel_fmax)
        self.register_buffer('mel_filter',
                             torch.tensor(mel_filters, dtype=torch.float))

        # STFT
        self.stft = STFT(n_fft, hop_len)

    def forward(self, x, out_spec=False):
        # do STFT
        z = self.stft(x)
        # get magnitude
        mag, _ = STFT.to_mag_phase(z)
        # apply mel filter
        mel = torch.matmul(self.mel_filter, mag)
        # minimum mel value
        min_mel = np.power(10, self.min_db / 10)
        # to log-space
        mel = torch.log(mel.clamp(min=min_mel))
        # return results
        if out_spec:
            # to log-space
            mag = torch.log(mag.clamp(min=min_mel))
            return mel, mag
        else:
            return mel


class ISTFT(torch.nn.Module):

    def __init__(self, filter_len, hop_len):
        super().__init__()
        self.filter_len = filter_len
        self.hop_len = hop_len
        # save basis as buffer
        self.register_buffer('basis',
                             torch.tensor(_get_inverse_basis(filter_len, hop_len),
                                          dtype=torch.float).unsqueeze(1))
        # save sum square window as buffer
        self.register_buffer('square_win',
                             torch.tensor(_get_window(self.filter_len) ** 2,
                                          dtype=torch.float))

    def forward(self, stft):

        # do ISTFT by 1D transposed conv
        x = torch.conv_transpose1d(
            stft, self.basis, stride=self.hop_len, padding=0)

        # remove effect by HANN windowing
        x = self.remove_window_effect(x, stft.shape[2])

        # cut off padded area
        cut = self.filter_len // 2
        return x[:, :, cut:-cut].squeeze(1)

    def remove_window_effect(self, x, frames):

        # zero filter
        f = torch.zeros_like(x)[0, 0].fill_(1e-10)

        # Fill the envelope
        # TODO: re-building filling process
        for i in range(frames):
            sample = i * self.hop_len
            f[sample:min(len(f), sample + self.filter_len)] \
                += self.square_win[:max(0, min(self.filter_len, len(f) - sample))]

        # remove modulation effects
        x = x / f

        # scale by hop ratio
        x *= float(self.filter_len) / self.hop_len

        return x
