import numpy as np
import torch
from scipy.io.wavfile import read as wav_read
from typing import List
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from pytorch_sound.utils.sound import parse_midi
from pytorch_sound.utils.text import eng_t2i
from pytorch_sound.utils.tensor import fix_length
from pytorch_sound.data.meta import MetaFrame, MetaType


class SpeechDataset(Dataset):

    def __init__(self, meta_frame: MetaFrame, fix_len: int = 0, skip_audio: bool = False):
        """
        :param meta_frame: Data Frame with dataset info
        :param kwargs: attributes to load data
        """
        self.meta_frame = meta_frame
        self.fix_len = fix_len
        self.cols = self.meta_frame.process_columns
        if skip_audio:
            self.cols = [x for x in self.cols if x != MetaType.audio_filename.name]

    def __getitem__(self, idx: int):
        meta_item = self.meta_frame.iloc[idx]
        return self.handle_fields(meta_item)

    def handle_fields(self, meta_item) -> List:
        results = []
        for col in self.meta_frame.process_columns:
            if col == MetaType.audio_filename.name:
                item = self.load_audio(meta_item[col])
            elif col == MetaType.midi_filename.name:
                item = self.load_midi(meta_item[col])
            elif col == MetaType.speaker.name:
                item = int(meta_item[col])
            elif col == MetaType.text.name:
                item = self.load_txt(meta_item[col])
            else:
                raise NotImplementedError('{} is not implemented !'.format(col.value))
            results.append(item)
        return results

    def load_audio(self, file_path: str) -> List[np.ndarray]:
        sr, wav = wav_read(file_path)
        assert sr == self.meta_frame.sr, \
            'sample rate miss match.\n {}\t {} in {}'.format(self.meta_frame.sr, sr, file_path)
        # random crop
        if self.fix_len:
            start_idx = np.random.randint(0, max(1, len(wav) - self.fix_len + 1))
            wav = fix_length(wav[start_idx:], self.fix_len)
        return [wav]

    @staticmethod
    def load_midi(file_path: str) -> List[np.ndarray]:
        """
        :param file_path: midi file path
        :return: piano roll with default setting
        """
        # load midi file
        mid = parse_midi(file_path)
        # TODO: enhance preprocess midi info
        return [mid.get_piano_roll()]

    @staticmethod
    def load_txt(txt: str) -> List[int]:
        return eng_t2i(txt)

    def __len__(self) -> int:
        return len(self.meta_frame)


class SpeechDataLoader(DataLoader):

    def __init__(self, dataset: SpeechDataset, batch_size: int, num_workers: int, is_bucket: bool):
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


if __name__ == '__main__':
    import sys
    from pytorch_sound.data.meta.libri_tts import LibriTTSMeta

    args = sys.argv[1:]
    meta_path = args[0]
    meta = LibriTTSMeta(meta_path)
    dataset = SpeechDataset(meta)
    print(len(dataset))
