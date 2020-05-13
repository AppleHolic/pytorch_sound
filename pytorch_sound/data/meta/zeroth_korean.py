import pandas as pd
import os
from typing import List, Tuple, Callable

from pytorch_sound import settings
from pytorch_sound.data.dataset import SpeechDataLoader, SpeechDataset
from pytorch_sound.data.korean.g2p import KoG2P
from pytorch_sound.data.meta import MetaFrame, MetaType
from pytorch_sound.data.meta.commons import split_train_val_frame


class ZerothKoreanMeta(MetaFrame):
    """
    Extended MetaFrame for zeroth korean
    - dataset : http://openslr.org/40
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
        return [(MetaType.AUDIO, 'audio_filename'), (MetaType.META, 'duration'), (MetaType.TEXT, 'text'),
                (MetaType.TEXT, 'phoneme'), (MetaType.SCALAR, 'speaker')]

    @property
    def meta(self) -> pd.DataFrame:
        return self._meta

    def __len__(self):
        return len(self._meta)

    def make_meta(self, wav_file_list: List[str], text_file_list: List[str]):
        # make mapper
        info = {'audio_filename': {}, 'text': {}, 'phoneme': {}, 'speaker': {}}
        speakers = []

        # parse wav keys
        for wav_path in wav_file_list:
            key = os.path.basename(wav_path).replace('.wav', '')
            info['audio_filename'][key] = wav_path
            speaker = key.split('_')[0]
            speakers.append(speaker)

        # speaker
        speakers = {spk: idx for idx, spk in enumerate(list(set(speakers)))}
        for key in info['audio_filename'].keys():
            speaker = key.split('_')[0]
            info['speaker'][key] = speakers[speaker]

        # parse texts
        # kog2p
        kog2p = KoG2P()
        for text_path in text_file_list:
            # read text file
            with open(text_path, 'r') as r:
                for line in r.readlines():
                    spl = line.split(' ')
                    key, text = spl[0], ' '.join(spl[1:])
                    info['text'][key] = text
                    info['phoneme'][key] = kog2p.g2p(text)

        # change meta obj
        self._meta = pd.DataFrame(info)
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
        train_meta, val_meta = split_train_val_frame(self._meta, val_rate=0.05)

        # save data frames
        print('Save meta frames on {}'.format(' '.join(self.frame_file_names)))
        self.save_meta(self.frame_file_names, self.meta_path, self._meta, train_meta, val_meta)


def check_phn_dict(meta_path: str):
    # read json meta file
    df = pd.read_json(meta_path)

    # parse phones
    phoneme_list = df['phoneme'].values
    phoneme_set = set([phn for phns in phoneme_list for phn in phns.split()])

    print('\n'.join(sorted(list(phoneme_set))))


if __name__ == '__main__':
    import sys
    p = sys.argv[1]
    check_phn_dict(p)
