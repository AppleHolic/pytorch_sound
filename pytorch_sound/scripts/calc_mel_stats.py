import fire
import torch
import os

from tqdm import tqdm
from pytorch_sound import settings
from pytorch_sound.data.meta.libri_tts import get_datasets
from pytorch_sound.models.stft import MelSpectrogram
from pytorch_sound.utils.calculate import volume_norm_log_torch
from pytorch_sound.utils.tensor import to_device


def main(meta_dir: str, out_path: str = '', batch_size: int = 32, num_workers: int = 8):
    # make mel transform inst
    mel_trans = MelSpectrogram(settings.SAMPLE_RATE, settings.WIN_LENGTH, settings.HOP_LENGTH,
                               settings.MEL_SIZE, settings.MEL_MIN, settings.MEL_MAX, settings.MIN_DB).cuda()

    # get data loader
    train_loader, _ = get_datasets(meta_dir, batch_size=batch_size, num_workers=num_workers)

    # make buffer
    mel_buf = torch.zeros(settings.MEL_SIZE).cuda()
    total_size = 0

    # calc loop
    for dp in tqdm(train_loader):
        # to device(cuda)
        wav, _, _ = to_device(dp)

        # volumn normalization
        wav = volume_norm_log_torch(wav).clamp(-1., 1.)  # (N, C, T)

        # to mel
        mel = mel_trans(wav).data

        # reduce
        mean_mel = mel.mean(dim=2).mean(dim=0)  # (C,)

        # add to buffer
        mel_buf += mean_mel

        # add size
        total_size += mel.size(0)

    # to cpu, list
    mel_buf /= total_size
    mel_f_list = mel_buf.cpu().numpy().tolist()

    # print and write
    if out_path:
        os.makedirs(os.path.split(out_path)[0], exist_ok=True)
        with open(out_path, 'w') as w:
            w.write('\n'.join(list(map(str, mel_f_list))))


if __name__ == '__main__':
    fire.Fire(main)
