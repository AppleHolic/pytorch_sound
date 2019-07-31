import pandas as pd
import os
import glob
from typing import List, Tuple, Callable
from tqdm import tqdm
from itertools import repeat

from pytorch_sound.data.dataset import SpeechDataset, SpeechDataLoader
from pytorch_sound.data.meta import MetaFrame, MetaType
from pytorch_sound.data.meta.commons import split_train_val_frame


class LibriTTSMeta(MetaFrame):
    """
    Extended MetaFrame for using LibriTTS
    - dataset : http://www.openslr.org/60/
    - arxiv : https://arxiv.org/abs/1904.02882
    """
    frame_file_names: List[str] = ['all_meta.json', 'train_meta.json', 'val_meta.json']

    def __init__(self, meta_path: str = ''):
        self.meta_path = meta_path
        if os.path.exists(self.meta_path) and not os.path.isdir(self.meta_path):
            self._meta = pd.read_json(self.meta_path)
            self._meta = self._meta.sort_values(by='duration')
        else:
            self._meta = pd.DataFrame(columns=self.columns, data={})
        # setup parameters
        self._num_speakers = None

    @property
    def columns(self) -> List[Tuple[MetaType, str]]:
        return [(MetaType.AUDIO, 'audio_filename'), (MetaType.SCALAR, 'speaker'),
                (MetaType.META, 'duration'), (MetaType.TEXT, 'text')]

    @property
    def meta(self) -> pd.DataFrame:
        return self._meta

    @property
    def sr(self) -> int:
        return 22050

    @property
    def num_speakers(self):
        if self._num_speakers is None:
            speakers = self._meta['speaker'].values
            set_speakers = set(speakers)
            self._num_speakers = len(set_speakers)
        return self._num_speakers

    def __len__(self):
        return len(self._meta)

    #
    # preprocess functions
    #
    def make_meta(self, root_dir: str, min_wav_rate: int, max_wav_rate: int, min_txt_rate: float):
        # speakers
        print('list up speakers')
        speakers = os.listdir(root_dir)

        # look up files
        print('lookup files...')
        wav_file_list = []
        speaker_mult = []
        for speaker in tqdm(speakers):
            file_temp = glob.glob(os.path.join(root_dir, speaker, 'wav', '*.wav'))
            wav_file_list.extend(file_temp)
            speaker_mult.extend(list(repeat(speaker, len(file_temp))))

        print('Update meta infos')
        speaker_mappings = {spk: idx for idx, spk in enumerate(sorted(speakers))}
        # update infos
        self._meta['speaker'] = [speaker_mappings[idx] for idx in speaker_mult]
        self._meta['audio_filename'] = wav_file_list
        self._meta['pass'] = [True] * len(speaker_mult)

        # read duration
        print('Check durations on wave files ...')
        dur_list = self._process_duration(wav_file_list, min_wav_rate, max_wav_rate)
        self._meta['duration'] = dur_list

        # text process
        print('Text pre-process ... ')
        txt_file_list = [file_path.replace('wav', 'txt') for file_path in wav_file_list]
        self._process_txt(txt_file_list, dur_list, min_txt_rate)

        # filter passed rows
        self._meta = self._meta[self._meta['pass'].values]

        # split train / val
        print('Make train / val meta')
        train_meta, val_meta = split_train_val_frame(self._meta)

        # save data frames
        print('Save meta frames on {}'.format(' '.join(self.frame_file_names)))
        self.save_meta(self.frame_file_names, self.meta_path, self._meta, train_meta, val_meta)


def get_datasets(meta_dir: str, batch_size: int, num_workers: int,
                 fix_len: int = 0, skip_audio: bool = False,
                 audio_mask: bool = False, skip_last_bucket: bool = True,
                 extra_features: List[Tuple[str, Callable]] = None) -> Tuple[SpeechDataLoader, SpeechDataLoader]:
    # TODO: update this function in general
    assert os.path.isdir(meta_dir), '{} is not valid directory path!'

    train_file, valid_file = LibriTTSMeta.frame_file_names[1:]

    # load meta file
    train_meta = LibriTTSMeta(os.path.join(meta_dir, train_file))
    valid_meta = LibriTTSMeta(os.path.join(meta_dir, valid_file))

    # create dataset
    train_dataset = SpeechDataset(train_meta, fix_len=fix_len, skip_audio=skip_audio, audio_mask=audio_mask,
                                  extra_features=extra_features)
    valid_dataset = SpeechDataset(valid_meta, fix_len=fix_len, skip_audio=skip_audio, audio_mask=audio_mask,
                                  extra_features=extra_features)

    # create data loader
    train_loader = SpeechDataLoader(train_dataset, batch_size=batch_size,
                                    num_workers=num_workers, skip_last_bucket=skip_last_bucket)
    valid_loader = SpeechDataLoader(valid_dataset, batch_size=batch_size, is_bucket=False, num_workers=num_workers)

    return train_loader, valid_loader


def get_speakers(meta_dir: str) -> int:
    """
    the number of speakers in libri-tts trainset
    :param meta_dir: meta directory path
    :return: the number of speakers
    """
    assert os.path.isdir(meta_dir), '{} is not valid directory path!'

    train_file = LibriTTSMeta.frame_file_names[1]

    # load meta file
    train_meta = LibriTTSMeta(os.path.join(meta_dir, train_file))

    return train_meta.num_speakers


if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
    root_dir, meta_path = args[:2]
    min_wav_rate, max_wav_rate, min_txt_rate = list(map(float, args[2:]))
    meta = LibriTTSMeta(meta_path)
    meta.make_meta(root_dir, min_wav_rate, max_wav_rate, min_txt_rate)
