import pandas as pd
import os
from typing import List, Tuple
from pytorch_sound.data.dataset import SpeechDataLoader
from pytorch_sound.data.meta import MetaFrame, MetaType


class MedleyDBMeta(MetaFrame):
    """
    Extended MetaFrame for using MedleyDB
    MedleyDB is the dataset has both seperated sound sources and mixture of them.
    Referece Link : https://github.com/marl/medleydb
    """
    frame_file_names: List[str] = ['all_meta.json', 'train_meta.json', 'val_meta.json']

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
        raise NotImplementedError('Meta file is not described yet !')


def get_datasets(meta_dir: str, batch_size: int, num_workers: int,
                 fix_len: int = 0, audio_mask: bool = False) -> Tuple[SpeechDataLoader, SpeechDataLoader]:
    raise NotImplementedError('To be coded')
