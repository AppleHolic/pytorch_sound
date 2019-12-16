import pandas as pd
import os
from typing import List, Tuple, Callable

from pytorch_sound import settings
from pytorch_sound.data.dataset import SpeechDataLoader, SpeechDataset
from pytorch_sound.data.meta import MetaFrame, MetaType
from pytorch_sound.data.meta.commons import split_train_val_frame


class LJSpeechMeta(MetaFrame):
    """
    Extended MetaFrame for LJSpeech Dataset
    - dataset : https://keithito.com/LJ-Speech-Dataset/
    """
    frame_file_names: List[str] = ['all_meta.json', 'train_meta.json', 'val_meta.json']

    def __init__(self, meta_path: str = '', sr: int = 22050):
        self.meta_path = meta_path
        if os.path.exists(self.meta_path) and not os.path.isdir(self.meta_path):
            self._meta = pd.read_json(self.meta_path)
            self._meta = self._meta.sort_values(by='duration')
        else:
            self._meta = pd.DataFrame(columns=self.column_names, data={})
        # setup parameters
        self._num_speakers = None
        self.sr = sr

    @property
    def columns(self) -> List[Tuple[MetaType, str]]:
        return [(MetaType.AUDIO, 'audio_filename'), (MetaType.META, 'duration'), (MetaType.TEXT, 'text')]

    @property
    def meta(self) -> pd.DataFrame:
        return self._meta

    def __len__(self):
        return len(self._meta)

    def make_meta(self, wav_file_list: List[str], txt_info: pd.DataFrame):

        # dictionary for making data frame
        infos = {'audio_filename': {}, 'text': {}}

        # make wav file / id mapping
        wav_mapping = {os.path.basename(wav_file_path).split('.')[0]: wav_file_path for wav_file_path in wav_file_list}

        # mapping texts
        for row in txt_info.iterrows():
            val_series = row[1]
            # ['id', 'text', 'normalized_text']
            id_, norm_text = val_series['id'], val_series['normalized_text']

            wav_file_path = wav_mapping[id_]

            infos['audio_filename'][id_] = wav_file_path
            infos['text'][id_] = norm_text

        # change meta obj
        self._meta = pd.DataFrame(infos)
        self._meta['pass'] = [True] * len(self._meta)

        # read duration
        print('Check durations on wave files ...')
        dur_list = self._process_duration(self._meta['audio_filename'].values,
                                          settings.MIN_WAV_RATE, settings.MAX_WAV_RATE)
        self._meta['duration'] = dur_list

        # filter passed rows
        self._meta = self._meta[self._meta['pass'].values]
        self._meta = self._meta.dropna()

        # split train / val
        print('Make train / val meta')
        train_meta, val_meta = split_train_val_frame(self._meta, val_rate=0.1)

        # save data frames
        print('Save meta frames on {}'.format(' '.join(self.frame_file_names)))
        self.save_meta(self.frame_file_names, self.meta_path, self._meta, train_meta, val_meta)


def get_datasets(meta_dir: str, batch_size: int, num_workers: int,
                 fix_len: int = 0, skip_audio: bool = False,
                 audio_mask: bool = False,
                 extra_features: List[Tuple[str, Callable]] = None) -> Tuple[SpeechDataLoader, SpeechDataLoader]:
    # TODO: update this function in general
    assert os.path.isdir(meta_dir), '{} is not valid directory path!'

    train_file, valid_file = LJSpeechMeta.frame_file_names[1:]

    # load meta file
    train_meta = LJSpeechMeta(os.path.join(meta_dir, train_file))
    valid_meta = LJSpeechMeta(os.path.join(meta_dir, valid_file))

    # create dataset
    train_dataset = SpeechDataset(train_meta, fix_len=fix_len, skip_audio=skip_audio, audio_mask=audio_mask,
                                  extra_features=extra_features)
    valid_dataset = SpeechDataset(valid_meta, fix_len=fix_len, skip_audio=skip_audio, audio_mask=audio_mask,
                                  extra_features=extra_features)

    # create data loader
    train_loader = SpeechDataLoader(train_dataset, batch_size=batch_size, is_bucket=True, num_workers=num_workers, n_buckets=5)
    valid_loader = SpeechDataLoader(valid_dataset, batch_size=batch_size, is_bucket=False, num_workers=num_workers)

    return train_loader, valid_loader


if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
    root_dir, meta_path = args[:2]
    meta = LJSpeechMeta(meta_path)
    meta.make_meta(root_dir)
