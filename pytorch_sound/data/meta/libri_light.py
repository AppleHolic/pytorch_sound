import pandas as pd
import os
import json

from typing import List, Tuple

from pytorch_sound.data.meta.commons import split_train_val_frame

from pytorch_sound.data.dataset import SpeechDataLoader, SpeechDataset
from pytorch_sound.data.meta import MetaFrame, MetaType


class LibriLightMeta(MetaFrame):
    """
    Extended MetaFrame for using Libri Light Dataset
    https://github.com/facebookresearch/libri-light
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
        return [(MetaType.AUDIO, 'audio_filename'), (MetaType.SCALAR, 'speaker'), (MetaType.META, 'duration')]

    @property
    def meta(self) -> pd.DataFrame:
        return self._meta

    @property
    def num_speakers(self):
        if self._num_speakers is None:
            speakers = self._meta['speaker'].values
            set_speakers = set(speakers)
            self._num_speakers = len(set_speakers)
        return self._num_speakers

    def __len__(self):
        return len(self._meta)

    def make_meta(self, chunk_file_list, speakers, val_rate: float = 0.1):
        # make dict
        info = {'audio_filename': chunk_file_list, 'speaker': speakers}

        # change meta obj
        self._meta = pd.DataFrame(info)

        # make speaker as indices
        speaker_mappings = {spk: idx for idx, spk in enumerate(sorted(list(set(self._meta['speaker'].values))))}

        # update infos
        self._meta['speaker'] = [speaker_mappings[spk] for spk in self._meta['speaker'].values]
        self._meta['pass'] = [True] * len(self._meta)

        # read duration
        print('Check durations on wave files ...')
        dur_list = self._process_duration(self._meta['audio_filename'].values, 0, 0)
        self._meta['duration'] = dur_list

        # split train / val
        print('Make train / val meta')
        train_meta, val_meta = split_train_val_frame(self._meta, val_rate=val_rate)

        # save data frames
        print('Save meta frames on {}'.format(' '.join(self.frame_file_names)))
        self.save_meta(self.frame_file_names, self.meta_path, self._meta, train_meta, val_meta)

        # save speaker map as json
        spk_json_path = os.path.join(self.meta_path, 'speaker_map.json')
        with open(spk_json_path, 'w') as w:
            json.dump(speaker_mappings, w)


def get_datasets(meta_dir: str, batch_size: int, num_workers: int,
                 fix_len: int = 0, skip_audio: bool = False,
                 audio_mask: bool = False) -> Tuple[SpeechDataLoader, SpeechDataLoader]:
    # TODO: update this function in general
    assert os.path.isdir(meta_dir), '{} is not valid directory path!'

    train_file, valid_file = LibriLightMeta.frame_file_names[1:]

    # load meta file
    train_meta = LibriLightMeta(os.path.join(meta_dir, train_file))
    valid_meta = LibriLightMeta(os.path.join(meta_dir, valid_file))

    # create dataset
    train_dataset = SpeechDataset(train_meta, fix_len=fix_len, skip_audio=skip_audio, audio_mask=audio_mask)
    valid_dataset = SpeechDataset(valid_meta, fix_len=fix_len, skip_audio=skip_audio, audio_mask=audio_mask)

    # create data loader
    train_loader = SpeechDataLoader(train_dataset, batch_size=batch_size,
                                    num_workers=num_workers, skip_last_bucket=False)
    valid_loader = SpeechDataLoader(valid_dataset, batch_size=batch_size,
                                    num_workers=num_workers, skip_last_bucket=False)

    return train_loader, valid_loader
