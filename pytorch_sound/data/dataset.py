import numpy as np
import torch
import os
from scipy.io.wavfile import read as wav_read
from pytorch_sound.utils.text import eng_t2i
from pytorch_sound.utils.tensor import fix_length
from pytorch_sound.settings import DataType
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate


class SpeechDataset(Dataset):

    def __init__(self, root_path: str, data_type: DataType, is_train: bool=False):
        # args
        self.data_type = data_type
        self.is_train = is_train
        self.sr = 0

        # path setting
        assert os.path.exists(root_path) and os.path.isdir(root_path), '{} is not valid path!'.format(root_path)

        # TODO: re-build meta info, dataset

    def __getitem__(self, idx):
        return {
            DataType.WAV: self.handle_wav,
            DataType.ENG_WAV: self.handle_eng_wav,
        }[self.data_type](idx)

    def handle_wav(self, idx):
        # vanilla wav
        wav = self.load_wav(idx)
        # random crop
        fix_len = self.hparam.get('fix_len', 0)
        if fix_len:
            sidx = np.random.randint(0, max(1, len(wav) - fix_len + 1))
            wav = fix_length(wav[sidx:], fix_len)
        return [wav]

    def handle_eng_wav(self, idx):
        wav = self.load_wav(idx)
        script = self.load_eng(idx)
        return script, wav, int(self.meta_info[idx][0])

    def load_wav(self, idx):
        wav_path = self.get_path(idx, 'wav')
        sr, wav = wav_read(wav_path)
        # TOOD: sample rate rebuild
        return wav

    def load_eng(self, idx):
        with open(self.get_path(idx, 'txt'), 'r', encoding='utf-8') as f:
            ixs = eng_t2i(f.read())
        return ixs

    def __len__(self):
        return len(self.meta_info)


class SpeechDataLoader(DataLoader):

    def __init__(self, dataset: Dataset, batch_size: int, num_workers: int, is_bucket: bool):
        # call super
        super().__init__(dataset,
                         num_workers=num_workers,
                         collate_fn=self.pad_collate_fn,
                         pin_memory=True,
                         batch_size=(1 if is_bucket else batch_size),
                         shuffle=(not is_bucket))

    @staticmethod
    def pad_collate_fn(batch):

        if len(batch) > 1:
            # do zero-padding
            result = []
            for i in range(len(batch[0])):
                # apply padding on dataset
                sub_batch = [x[i] for x in batch]
                # check diff dims
                if not isinstance(sub_batch[0], np.ndarray):
                    # if list of float or int
                    assert all([type(x) == type(sub_batch[0]) for x in sub_batch[1:]])
                    if isinstance(sub_batch[0], int):
                        sub_batch = torch.LongTensor(sub_batch)
                    elif isinstance(batch[0], float):
                        sub_batch = torch.DoubleTensor(sub_batch)

                elif any(list(map(lambda x: x.shape != sub_batch[0].shape, sub_batch[1:]))):
                    sub_batch = torch.from_numpy(__class__.__pad_zero(sub_batch))
                else:
                    sub_batch = torch.from_numpy(np.concatenate(np.expand_dims(sub_batch, axis=0)))
                result.append(sub_batch)
            return result
        else:
            if None in batch:
                return None
            else:
                return default_collate(batch)

    @staticmethod
    def __pad_zero(sub_batch):
        dims = [b.shape for b in sub_batch]

        max_dims = list(dims[0])
        for d_li in dims[1:]:
            for d_idx in range(len(d_li)):
                if max_dims[d_idx] < d_li[d_idx]:
                    max_dims[d_idx] = d_li[d_idx]

        temp = np.zeros((len(sub_batch), *max_dims), dtype=sub_batch[0].dtype)
        for i, b in enumerate(sub_batch):
            if len(b.shape) == 1:
                temp[i, :b.shape[0]] = b
            elif len(b.shape) == 2:
                temp[i, :b.shape[0], :b.shape[1]] = b
            elif len(b.shape) == 3:
                temp[i, :b.shape[0], :b.shape[1], :b.shape[2]] = b
            else:
                raise ValueError
        return temp
