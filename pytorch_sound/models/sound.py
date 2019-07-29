import torch


class VolNormWindow:

    def __init__(self, window_size: int, target_db: float):
        self.window_size = window_size
        self.target_db = target_db
        self.prev_wav_len = -1
        # make std buffer for reverse norm
        self.std_buffer = None

    def init_buffer(self, wav_len: int):
        self.prev_wav_len = wav_len
        self.std_buffer = torch.zeros(wav_len // self.window_size)

    def forward(self, wav: torch.tensor) -> torch.tensor:
        wav_len = wav.size(-1)
        self.init_buffer(wav_len)

        norm_wav_chunks = []

        for idx in range(wav_len // self.window_size):
            if idx < wav_len // self.window_size - 1:
                wav_slice = wav[..., self.window_size * idx:self.window_size * (idx + 1)]
            else:
                wav_slice = wav[..., self.window_size * idx:]

            wav_std = torch.std(wav_slice)
            self.std_buffer[idx:idx+1] = wav_std.squeeze()
            wav_norm_slice = wav_slice / (wav_std / 10 ** (self.target_db / 10))
            norm_wav_chunks.append(wav_norm_slice)

        return torch.cat(norm_wav_chunks, dim=-1)

    def reverse(self, wav: torch.tensor) -> torch.tensor:
        wav_len = wav.size(-1)
        assert self.prev_wav_len == wav_len

        wav_chunks = []

        for idx in range(wav_len // self.window_size):

            if idx < wav_len // self.window_size - 1:
                wav_slice = wav[..., self.window_size * idx:self.window_size * (idx + 1)]
            else:
                wav_slice = wav[..., self.window_size * idx:]

            wav_std = self.std_buffer[idx]
            unnorm_wav = wav_slice * (wav_std / 10 ** (self.target_db / 10))
            wav_chunks.append(unnorm_wav)

        return torch.cat(wav_chunks, dim=-1)
