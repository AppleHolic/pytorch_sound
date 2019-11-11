import torch
import torch.nn as nn
import librosa
from torchaudio.functional import istft
from torchaudio.transforms import AmplitudeToDB


class STFT(nn.Module):
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

    def transform(self, wav: torch.Tensor) -> torch.Tensor:
        """
        :param wav: wave tensor
        :return: (N, Spec Dimension * 2, T) 3 dimensional stft tensor
        """
        stft = torch.stft(
            wav, self.n_fft, self.hop_length, self.win_length, self.window, True,
            'reflect', False, True
        )  # (N, C, T, 2)
        real_part, img_part = [x.squeeze(3) for x in stft.chunk(2, 3)]
        return torch.sqrt(real_part ** 2 + img_part ** 2), torch.atan2(img_part, real_part)

    def inverse(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        # match dimension
        magnitude, phase = magnitude.unsqueeze(3), phase.unsqueeze(3)
        stft = torch.cat([magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=3)
        return istft(
            stft, self.n_fft, self.hop_length, self.win_length, self.window
        )


class LogMelSpectrogram(nn.Module):
    """
    Mel spectrogram module with above STFT class
    """

    def __init__(self, sample_rate: int, mel_size: int, n_fft: int, win_length: int,
                 hop_length: int, min_db: float, max_db: float,
                 mel_min: float = 0., mel_max: float = None):
        super().__init__()
        self.mel_size = mel_size
        self.min_db = min_db
        self.max_db = max_db

        # make stft func
        self.stft = STFT(filter_length=win_length, hop_length=hop_length)

        # mel filter banks
        mel_filter = librosa.filters.mel(sample_rate, n_fft, mel_size, fmin=mel_min, fmax=mel_max)
        self.register_buffer('mel_filter', torch.tensor(mel_filter, dtype=torch.float))

        # amp2db function
        self.amp2db = AmplitudeToDB()

    def forward(self, wav: torch.tensor, log_offset: float = 1e-6) -> torch.tensor:
        mag, phase = self.stft.transform(wav)

        # apply mel filter
        mel = torch.matmul(self.mel_filter, mag)

        # to db scale
        mel = self.amp2db(mel)

        # clip
        mel = mel.clamp(self.min_db, self.max_db)

        # to log-space
        mel = torch.log(mel + log_offset)

        return mel


class MelMasker(nn.Module):
    """
    Helper class transforming wave-level mask to spectrogram-level mask
    """

    def __init__(self, win_length: int, hop_length: int):
        super().__init__()
        self.conv = nn.Conv1d(
            1, 1, win_length, stride=hop_length, padding=win_length // 2, bias=False).cuda()
        torch.nn.init.constant_(self.conv.weight, 1.)

    def forward(self, wav_mask: torch.tensor, mask_val: float = 0.0) -> torch.tensor:
        # make mask
        with torch.no_grad():
            mel_mask = self.conv(wav_mask.float().unsqueeze(1)).squeeze(1)
            mel_mask = (mel_mask != mask_val).float()
        return mel_mask


class MelToMFCC(nn.Module):
    """
    Create the Mel-frequency cepstrum coefficients from mel-spectrogram
    """

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
    """
    Create the Mel-frequency cepstrum coefficients from an audio signal
    """

    def __init__(self, sample_rate: int, mel_size: int, n_fft: int, win_length: int, n_mfcc: int,
                 hop_length: int, min_db: float, max_db: float,
                 mel_min: float = 0., mel_max: float = None):
        super().__init__()
        self.n_mfcc = n_mfcc
        self.mel_func = LogMelSpectrogram(
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
