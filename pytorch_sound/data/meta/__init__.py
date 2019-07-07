"""
Meta Information Building
"""
import enum
import abc
import os
import re
import pandas as pd
from typing import List, Tuple
from itertools import repeat
    

from pytorch_sound.data.meta.commons import get_wav_duration
from pytorch_sound.utils.commons import go_multiprocess


class MetaType(enum.Enum):
    AUDIO: int = 1
    SCALAR: int = 2
    MIDI: int = 3
    TEXT: int = 4
    META: int = 5


class MetaFrame:

    @property
    def process_columns(self) -> List[str]:
        target_types = [MetaType.AUDIO, MetaType.SCALAR, MetaType.MIDI, MetaType.TEXT]
        return [(type_, name) for (type_, name) in self.columns if type_ in target_types]

    @property
    @abc.abstractmethod
    def columns(self) -> List[Tuple[MetaType, str]]:
        raise NotImplementedError('You must define columns !')

    @property
    @abc.abstractmethod
    def meta(self) -> pd.DataFrame:
        raise NotImplementedError('You must define make DataFrame!')

    @abc.abstractmethod
    def make_meta(self):
        raise NotImplementedError('You must define make DataFrame and save it !')

    @property
    def iloc(self):
        return self.meta.iloc

    def _process_duration(self, wav_file_list: List[str], min_wav_rate: float, max_wav_rate: float) -> List[float]:
        dur_list = go_multiprocess(get_wav_duration, wav_file_list)
        # check pass
        pass_list = []
        for p, dur in zip(self._meta['pass'], dur_list):
            flag = p and dur != -1
            if min_wav_rate and max_wav_rate:
                flag = flag and min_wav_rate < dur < max_wav_rate
            pass_list.append(flag)

        self._meta['pass'] = pass_list
        return dur_list

    def _process_txt(self, txt_file_list: List[str], dur_list: List[float], min_txt_rate: float):
        # do txt process
        results = go_multiprocess(preprocess_text, list(zip(txt_file_list,
                                                            repeat(min_txt_rate, len(txt_file_list)), dur_list)))
        # split lists
        txt_list, pass_list = map(list, zip(*results))
        self._meta['pass'] = [p1 and p2 for p1, p2 in zip(self._meta['pass'], pass_list)]
        self._meta['text'] = txt_list

    @staticmethod
    def save_meta(frame_file_names: List[str], meta_path: str, 
                  all_frame: pd.DataFrame, train_frame: pd.DataFrame, val_frame: pd.DataFrame):
        assert not os.path.exists(meta_path) or os.path.isdir(meta_path)
        if not os.path.exists(meta_path):
            os.makedirs(meta_path)
        # make names
        file_paths = [os.path.join(meta_path, name) for name in frame_file_names]
        # save
        all_frame.to_json(file_paths[0])
        train_frame.to_json(file_paths[1])
        val_frame.to_json(file_paths[2])
        
        
def preprocess_text(args) -> List:

    txt_file, min_ratio, dur = args

    # compile regex
    regex = re.compile(r'[a-zA-Z\'\.\,\?\!\ ]+')

    try:
        with open(txt_file, encoding='utf-8') as r:
            txt = r.read().strip()

        txt = ' '.join(map(lambda x: x.strip(), regex.findall(txt)))
        txt_dur = len(' '.join(txt.split()))

        if min_ratio is None:
            _pass = True
        else:
            _pass = min_ratio <= (txt_dur / float(dur))

    except:
        txt, _pass = '', False

    return [txt, _pass]
