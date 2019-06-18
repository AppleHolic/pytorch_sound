import pandas as pd
import os
import glob
import re
from typing import List, Tuple
from tqdm import tqdm
from itertools import repeat

from pytorch_sound.data.dataset import SpeechDataset, SpeechDataLoader
from pytorch_sound.data.meta import MetaFrame
from pytorch_sound.data.meta.commons import go_multiprocess, split_train_val_frame, get_wav_duration


class LibriTTSMeta(MetaFrame):

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
    def columns(self) -> List[str]:
        return ['audio_filename', 'speaker', 'duration', 'text']

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
        self.save_meta(self.meta_path, self._meta, train_meta, val_meta)


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


def get_datasets(meta_dir: str, batch_size: int, num_workers: int,
                 fix_len: int = 0, skip_audio: bool = False,
                 audio_mask: bool = False) -> Tuple[SpeechDataLoader, SpeechDataLoader]:

    assert os.path.isdir(meta_dir), '{} is not valid directory path!'

    train_file, valid_file = LibriTTSMeta.frame_file_names[1:]

    # load meta file
    train_meta = LibriTTSMeta(os.path.join(meta_dir, train_file))
    valid_meta = LibriTTSMeta(os.path.join(meta_dir, valid_file))

    # create dataset
    train_dataset = SpeechDataset(train_meta, fix_len=fix_len, skip_audio=skip_audio, audio_mask=audio_mask)
    valid_dataset = SpeechDataset(valid_meta, fix_len=fix_len, skip_audio=skip_audio, audio_mask=audio_mask)

    # create data loader
    train_loader = SpeechDataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    valid_loader = SpeechDataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers)

    return train_loader, valid_loader


def get_speakers(meta_dir: str) -> int:
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
