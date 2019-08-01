import math
import numpy as np
import torch
import librosa
import copy
from typing import List, Tuple, Callable, Any
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.dataloader import default_collate
from pytorch_sound.utils.sound import parse_midi
from pytorch_sound.utils.text import eng_t2i
from pytorch_sound.data.meta import MetaFrame, MetaType


class SpeechDataset(Dataset):
    """
    General pytorch dataset class using MetaFrame.
    It focuses on removing repetitive implementing same dataset on each experiments.
    Because of the columns of meta frame are generalized data types, dataset class all can be generalized.

    # TODO: Add usage sample
    """

    def __init__(self, meta_frame: MetaFrame, fix_len: int = 0, fix_shuffle: bool = False,
                 skip_audio: bool = False, audio_mask: bool = False, extra_features: List[Tuple[str, Callable]] = None):
        """

        :param meta_frame: An instance of MetaFrame
        :param fix_len: fixing(random cropping) the length of wave data
        :param fix_shuffle: randomize or not choosing random start index in above
        :param skip_audio: skip loading waves
        :param audio_mask: add or not a masks of wave
        :param extra_features: extra features that calculated on specific column, it is added on last points.
        """
        self.meta_frame = meta_frame
        self.fix_len = fix_len
        self.fix_shuffle = fix_shuffle
        self.cols = self.meta_frame.process_columns
        self.audio_mask = audio_mask
        self.extra_features = extra_features

        if self.extra_features:
            column_names = [name for _, name in self.meta_frame.columns]
            assert all([name in column_names for name, _ in extra_features]), \
                'Unmatched extra_feature name! {} {}'.format(str(column_names), str(extra_features))
            self.target_idx_map = {name: idx for idx, (type_, name) in enumerate(self.meta_frame.process_columns)}

        if skip_audio:
            self.cols = [(t, name) for (t, name) in self.cols if t != MetaType.AUDIO]

    def __getitem__(self, idx: int) -> List:
        meta_item = self.meta_frame.iloc[idx]
        return self.handle_fields(meta_item)

    def handle_fields(self, meta_item) -> List:
        """
        Loading and processing data using meta information on meta frame
        :param meta_item: an row on meta frame
        :return: loaded data point
        """
        results = []
        mask = None
        start_idx = -1

        for col in self.meta_frame.process_columns:
            type_, name = col
            if type_ == MetaType.AUDIO:
                item = self.load_audio(meta_item[name])
                # random crop
                if self.fix_len:
                    if start_idx == -1 or self.fix_shuffle:
                        start_idx = np.random.randint(0, max(1, len(item) - self.fix_len + 1))
                    item = item[start_idx:start_idx + self.fix_len]
                if self.audio_mask and mask is None:
                    mask = np.ones_like(item)
            elif type_ == MetaType.MIDI:
                item = self.load_midi(meta_item[name])
            elif type_ == MetaType.SCALAR:
                item = int(meta_item[name])
            elif type_ == MetaType.TEXT:
                item = self.load_txt(meta_item[name])
            else:
                raise NotImplementedError('{} is not implemented !'.format(name))
            results.append(item)

        if self.extra_features:
            for ex in self.extra_features:
                name, func = ex
                item = results[self.target_idx_map[name]]
                ex_feature = func(item)
                results.append(ex_feature)

        if mask is not None:
            results.append(mask)

        return results

    def load_audio(self, file_path: str) -> np.ndarray:
        # Speed of librosa loading function is enhanced on version 0.7.0
        wav, sr = librosa.load(file_path, sr=None)
        assert sr == self.meta_frame.sr, \
            'sample rate miss match.\n {}\t {} in {}'.format(self.meta_frame.sr, sr, file_path)
        return wav

    @staticmethod
    def load_midi(file_path: str) -> List[np.ndarray]:
        """
        :param file_path: midi file path
        :return: piano roll with default setting
        """
        # load midi file
        mid = parse_midi(file_path)
        # TODO: enhance preprocess midi info
        return mid.get_piano_roll()

    @staticmethod
    def load_txt(txt: str) -> List[int]:
        return eng_t2i(txt)

    def __len__(self) -> int:
        return len(self.meta_frame)


class BucketRandomBatchSampler(Sampler):
    """
    Chunking samples into buckets and sample bucket id randomly for each mini batch.
    """

    def __init__(self, data_source: Dataset, n_buckets: int, batch_size: int, skip_last_bucket: bool = False):
        assert len(data_source) > n_buckets * batch_size, 'Data size is too small to use bucket sampler !'
        self.n_buckets = n_buckets
        self.data_size = len(data_source)
        self.batch_size = batch_size
        self.bucket_size = int(math.ceil(self.data_size / self.n_buckets))
        self.bucket_size -= self.bucket_size % batch_size

        if self.n_buckets <= 0:
            raise ValueError("the num of buckets has to be a positive value.")

        self.buckets = [list(range(i * self.bucket_size, (i + 1) * self.bucket_size))
                        for i in range(self.n_buckets - int(skip_last_bucket))]

    def __iter__(self):
        # copy buckets and shuffle indices
        buckets = copy.deepcopy(self.buckets)
        for idx in range(len(buckets)):
            np.random.shuffle(buckets[idx])

        # pop up indices
        while buckets:
            bucket_id = np.random.choice(range(len(buckets)))
            ids = buckets[bucket_id][-self.batch_size:]  # pick last
            buckets[bucket_id] = buckets[bucket_id][:-self.batch_size]
            if not buckets[bucket_id]:
                buckets.pop(bucket_id)
            yield ids

    def __len__(self):
        return self.bucket_size * self.n_buckets // self.batch_size


class SpeechDataLoader(DataLoader):
    """
    General data loader for loading speech related data.
    It has customized collate function to match lengths of data in each batches.

    # TODO: Add usage sample
    """

    def __init__(self, dataset: SpeechDataset, batch_size: int, num_workers: int,
                 n_buckets: int = 10, is_bucket: bool = True, skip_last_bucket: bool = False):

        batch_sampler = None
        if is_bucket:
            batch_sampler = BucketRandomBatchSampler(
                dataset, n_buckets=n_buckets, batch_size=batch_size, skip_last_bucket=skip_last_bucket)
        # call super
        super().__init__(dataset,
                         num_workers=num_workers,
                         collate_fn=self.pad_collate_fn,
                         pin_memory=True,
                         batch_size=(1 if is_bucket else batch_size),
                         shuffle=(not is_bucket),
                         batch_sampler=batch_sampler)

    @staticmethod
    def pad_collate_fn(batch: List[Any]) -> torch.tensor:
        """
        Matching lengths in using zero-pad
        :param batch: mini batch by sampled on dataset
        :return: collated tensor
        """
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
    def __pad_zero(sub_batch: List[np.ndarray]) -> np.ndarray:
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
