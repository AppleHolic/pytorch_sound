import torch
import torch.nn as nn
import os
from librosa.filters import mel
from pytorch_sound.models import build_model
from pytorch_sound.interface import Interface
from pytorch_sound.models.vocoders import hifi_gan


class AudioParameters:

    sampling_rate: int = 22050
    n_fft: int = 1024
    window_size: int = 1024
    hop_size: int = 256
    num_mels: int = 80
    fmin: float = 0.
    fmax: float = 8000.


CHKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'assets', 'vocoders')
CHECKPOINTS = {
    'hifi_gan_v1': os.path.join(CHKPT_DIR, 'hifi_gan_v1.pt'),
    'hifi_gan_v2': os.path.join(CHKPT_DIR, 'hifi_gan_v2.pt')
}


class MelSpectrogram(nn.Module):
    """
    Mel Spectrogram Module for using Hifi-GAN.
    - reference : https://github.com/jik876/hifi-gan/blob/master/meldataset.py
    """
    def __init__(self, sampling_rate: int = 22050, n_fft: int = 1024, window_size: int = 1024, hop_size: int = 256,
                 num_mels: int = 80, fmin: float = 0., fmax: float = 8000.):
        super().__init__()
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.window_size = window_size
        self.pad_size = (self.n_fft - self.hop_size) // 2

        mel_filter_tensor = torch.FloatTensor(mel(sampling_rate, n_fft, num_mels, fmin, fmax))
        self.register_buffer('mel_filter', mel_filter_tensor)
        self.register_buffer('window', torch.hann_window(window_size))

    def forward(self, wav: torch.Tensor, is_center: bool = False) -> torch.Tensor:
        # reflec padding
        wav = wav.unsqueeze(1)
        wav = torch.nn.functional.pad(wav, [self.pad_size, self.pad_size], mode='reflect').squeeze(1)

        # stft
        spec = torch.stft(wav, self.n_fft, hop_length=self.hop_size, win_length=self.window_size,
                          window=self.window, center=is_center, pad_mode='reflect',
                          normalized=False, onesided=True)
        spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)

        # mel
        spec = torch.matmul(self.mel_filter, spec)

        # spectral normalize
        spec = torch.log(torch.clamp(spec, min=1e-5) * 1.)

        return spec


class InterfaceHifiGAN(Interface):
    """
    An interface between wave and feature based speech generations models with Hifi-GAN
    You can select a model_name in ['hifi_gan_v1', 'hifi_gan_v2'].
      hifi_gan_v1 = higher memory and higher quality
      hifi_gan_v2 = lower memory and faster speed

    Examples::
        import torch
        import librosa
        from pytorch_sound.interface.hifi_gan import InterfaceHifiGAN

        # load interface
        interface = InterfaceHifiGAN()

        # load sample wav
        wav, sr = librosa.load('/your/wav/file', sr=22050)
        wav_tensor = torch.FloatTensor(wav).unsqueeze(0)

        # encode to mel
        mel = interface.encode(wav_tensor)

        # decode to wav using hifi-gan
        pred_wav = interface.decode(mel)
    """

    def __init__(self, model_name: str = 'hifi_gan_v1'):
        assert model_name in ['hifi_gan_v1', 'hifi_gan_v2'], \
            'Model name {} is not valid! choose in {}'.format(model_name, str(['hifi_gan_v1', 'hifi_gan_v2']))

        # encoder
        self.encoder = MelSpectrogram(**vars(AudioParameters()))

        # decoder
        self.decoder = build_model(model_name)
        chkpt = torch.load(CHECKPOINTS[model_name])
        self.decoder.load_state_dict(chkpt['generator'])
        self.decoder.remove_weight_norm()

    def encode(self, wav_tensor: torch.Tensor) -> torch.Tensor:
        assert wav_tensor.ndim == 2, '2D tensor (N, T) is needed'
        return self.encoder(wav_tensor)

    def decode(self, mel_tensor: torch.Tensor) -> torch.Tensor:
        assert mel_tensor.ndim == 3, '3D tensor (N, C, T) is needed'
        with torch.no_grad():
            return self.decoder(mel_tensor)
