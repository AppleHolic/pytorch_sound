import torch


class VolNormConv:
    """
    Enhancing volume normalization on sound by windowing normalization process.
    """

    def __init__(self, window_size: int, hop_size: int, target_db: float):
        self.window_size = window_size
        self.hop_size = hop_size
        self.target_db = target_db
        self.prev_wav_len = -1
        # make std buffer for reverse norm
        self.std_buffer = None

    def init_buffer(self, wav_len: int):
        self.prev_wav_len = wav_len
        self.std_buffer = torch.zeros((wav_len - self.window_size) // self.hop_size + 1)

    def forward(self, wav: torch.tensor) -> torch.tensor:
        wav_len = wav.size(-1)
        self.init_buffer(wav_len)

        norm_wav_chunks = []

        for idx, hop_point in enumerate(range(0, wav_len - self.window_size, self.hop_size)):
            if hop_point < wav_len - self.window_size - 1:
                wav_slice = wav.data[..., hop_point: hop_point + self.hop_size]
            else:
                wav_slice = wav.data[..., hop_point:]

            wav_std = torch.std(wav.data[..., hop_point: hop_point + self.window_size])
            self.std_buffer[idx] = wav_std
            wav_norm_slice = wav_slice / (wav_std / 10 ** (self.target_db / 10))
            norm_wav_chunks.append(wav_norm_slice)

        return torch.cat(norm_wav_chunks, dim=-1)

    def reverse(self, wav: torch.tensor) -> torch.tensor:
        wav_len = wav.size(-1)
        assert self.prev_wav_len >= wav_len, '{} is smaller than {} !'.format(self.prev_wav_len, wav_len)

        wav_chunks = []

        for idx, hop_point in enumerate(range(0, wav_len - self.window_size, self.hop_size)):

            if hop_point < wav_len - self.window_size - self.hop_size:
                wav_slice = wav.data[..., hop_point: hop_point + self.hop_size]
            else:
                wav_slice = wav.data[..., hop_point:]

            wav_std = self.std_buffer[idx]
            unnorm_wav = wav_slice * (wav_std / 10 ** (self.target_db / 10))
            wav_chunks.append(unnorm_wav)

        return torch.cat(wav_chunks, dim=-1)
