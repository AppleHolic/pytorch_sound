import pandas as pd
import os
import glob
from typing import List, Tuple
from pytorch_sound.data.dataset import SpeechDataLoader, SpeechDataset
from pytorch_sound.data.meta import MetaFrame, MetaType
from pytorch_sound.data.meta.commons import split_train_val_frame


class DSD100Meta(MetaFrame):
    """
    Extended MetaFrame for using DSD100
    - dataset : https://sigsep.github.io/datasets/dsd100.html
    """

    def __init__(self, meta_path: str = '', sr: int = 44100):
        self.meta_path = meta_path
        if os.path.exists(self.meta_path) and not os.path.isdir(self.meta_path):
            self._meta = pd.read_json(self.meta_path)
        else:
            self._meta = pd.DataFrame(columns=self.column_names, data={})
        self.sr = sr

    @property
    def columns(self) -> List[Tuple[MetaType, str]]:
        return [(MetaType.AUDIO, 'mixture_filename'), (MetaType.AUDIO, 'voice_filename')]

    @property
    def meta(self) -> pd.DataFrame:
        return self._meta

    def __len__(self):
        return len(self._meta)

    def make_meta(self, root_dir: str):
        # Use all audio files
        # directory names
        print('Lookup files ...')
        mixture_list = glob.glob(os.path.join(root_dir, 'Mixtures', '**', '**', 'mixture.*.npy'))

        # It only extract vocals. If you wanna use other source, override it.
        vocals_list = glob.glob(os.path.join(root_dir, 'Sources', '**', '**', 'vocals.*.npy'))

        # make meta dict
        print('Make meta information ...')

        # setup meta
        self._meta['mixture_filename'] = sorted(mixture_list)
        self._meta['voice_filename'] = sorted(vocals_list)

        # split train / val
        print('Make train / val meta')
        train_meta, val_meta = split_train_val_frame(self._meta, val_rate=0.1)

        # save data frames
        print('Save meta frames on {}'.format(' '.join(self.frame_file_names)))
        self.save_meta(self.frame_file_names, self.meta_path, self._meta, train_meta, val_meta)


def get_datasets(meta_dir: str, batch_size: int, num_workers: int,
                 fix_len: int = 0, audio_mask: bool = False) -> Tuple[SpeechDataLoader, SpeechDataLoader]:
    # TODO: update this function in general
    assert os.path.isdir(meta_dir), '{} is not valid directory path!'

    train_file, valid_file = DSD100Meta.frame_file_names[1:]

    # load meta file
    train_meta = DSD100Meta(os.path.join(meta_dir, train_file))
    valid_meta = DSD100Meta(os.path.join(meta_dir, valid_file))

    # create dataset
    train_dataset = SpeechDataset(train_meta, fix_len=fix_len, audio_mask=audio_mask)
    valid_dataset = SpeechDataset(valid_meta, fix_len=fix_len, audio_mask=audio_mask)

    # create data loader
    train_loader = SpeechDataLoader(train_dataset, batch_size=batch_size,
                                    num_workers=num_workers, is_bucket=False)
    valid_loader = SpeechDataLoader(valid_dataset, batch_size=batch_size,
                                    num_workers=num_workers, is_bucket=False)

    return train_loader, valid_loader
