import pandas as pd
import os
import glob
import re
from typing import Tuple, List
from tqdm import tqdm
from itertools import repeat
from pytorch_sound.data.meta import MetaFrame, MetaType
from pytorch_sound.data.meta.commons import go_multiprocess, split_train_val_frame
from pytorch_sound.utils.sound import get_wav_hdr


def preprocess_text(args) -> Tuple:

    txt_file, min_ratio, dur = args

    # compile regex
    regex = re.compile(r'[a-zA-Z\'\.\,\?\!\ ]+')

    try:
        with open(txt_file, encoding='utf-8') as r:
            txt = r.read().strip()

        txt = ' '.join(map(lambda x: x.strip(), regex.findall(txt)))
        txt_dur = len(' '.join(txt.split()))

        if min_ratio is None or min_ratio is None:
            _pass = True
        else:
            _pass = min_ratio <= txt_dur / float(dur) <= min_ratio

    except:
        txt, _pass = '', False

    return [txt, _pass]


def get_wav_duration(file_path: str) -> int:
    try:
        return get_wav_hdr(file_path)['Duration']
    except:
        return -1


class LibriTTSMeta(MetaFrame):

    def __init__(self, meta_path: str = '', min_wav_rate: float = 0.0, max_wav_rate: float = 0.0,
                 min_txt_rate: float = 0.0):
        self.meta_path = meta_path
        if os.path.exists(self.meta_path):
            self._meta = pd.read_json(self.meta_path)
        else:
            self._meta = pd.DataFrame(columns=self.columns, data={})
        # setup parameters
        self.min_wav_rate = min_wav_rate
        self.max_wav_rate = max_wav_rate
        self.min_txt_rate = min_txt_rate

    @property
    def columns(self):
        return [MetaType.audio_filename, MetaType.speaker, MetaType.duration, MetaType.text]

    @property
    def meta(self) -> pd.DataFrame:
        return self._meta

    @property
    def frame_file_names(self) -> List[str]:
        return ['all_meta.json', 'train_meta.json', 'val_meta.json']

    @property
    def sr(self) -> int:
        return 22050

    def process_duration(self, wav_file_list: List[str], min_wav_rate: float, max_wav_rate: float) -> List[float]:
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

    def process_txt(self, txt_file_list: List[str], dur_list: List[float]):
        # do txt process
        results = go_multiprocess(preprocess_text, list(zip(txt_file_list,
                                                            repeat(self.min_txt_rate, len(txt_file_list)), dur_list)))
        # split lists
        txt_list, pass_list = map(list, zip(*results))
        self._meta['pass'] = [p1 and p2 for p1, p2 in zip(self._meta['pass'], pass_list)]

    def save_meta(self, meta_path: str, all_frame: pd.DataFrame, train_frame: pd.DataFrame, val_frame: pd.DataFrame):
        assert not os.path.exists(meta_path) or os.path.isdir(meta_path)
        if not os.path.exists(meta_path):
            os.makedirs(meta_path)
        # make names
        file_paths = [os.path.join(meta_path, name) for name in self.frame_file_names]
        # save
        all_frame.to_json(file_paths[0])
        train_frame.to_json(file_paths[1])
        val_frame.to_json(file_paths[2])

    def make_meta(self, root_dir):
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
        # update infos
        self._meta['speaker'] = speaker_mult
        self._meta['file_path'] = wav_file_list
        self._meta['pass'] = [True] * len(speaker_mult)

        # read duration
        print('Check durations on wave files ...')
        dur_list = self.process_duration(wav_file_list, self.min_wav_rate, self.max_wav_rate)

        # text process
        print('Text pre-process ... ')
        txt_file_list = [file_path.replace('wav', 'txt') for file_path in wav_file_list]
        self.process_txt(txt_file_list, dur_list)

        # split train / val
        print('Make train / val meta')
        train_meta, val_meta = split_train_val_frame(self._meta)

        # save data frames
        print('Save meta frames on {}'.format(' '.join(self.frame_file_names)))
        self.save_meta(self.meta_path, self._meta, train_meta, val_meta)


if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
    root_dir, meta_path = args[:2]
    min_wav_rate, max_wav_rate, min_txt_rate = list(map(float, args[2:]))
    meta = LibriTTSMeta(meta_path, min_wav_rate, max_wav_rate, min_txt_rate)
    meta.make_meta(root_dir)
