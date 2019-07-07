import pandas as pd
import os
import glob

from tqdm import tqdm
from typing import List, Tuple
from collections import defaultdict

from pytorch_sound.data.dataset import SpeechDataLoader, SpeechDataset
from pytorch_sound.data.meta import MetaFrame, MetaType


class VoiceBankMeta(MetaFrame):

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
        return [(MetaType.AUDIO, 'noise_filename'), (MetaType.AUDIO, 'clean_filename'), (MetaType.SCALAR, 'speaker'),
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

    def make_meta(self, root_dir: str, min_wav_rate: int, max_wav_rate: int, min_txt_rate: float):
        info = defaultdict(dict)
        # composed by three keys
        # train or valid
        # ㄴ id
        # ㄴ ㄴ data type

        print('Lookup all files...')
        wav_file_list = glob.glob(os.path.join(root_dir, '**', '*.wav'))
        txt_file_list = glob.glob(os.path.join(root_dir, '**', '*.txt'))

        print('Match info structure')

        # match wave info
        for wav_file in tqdm(wav_file_list):
            key = os.path.basename(wav_file).replace('.wav', '')
            phase = 'train' if 'trainset' in wav_file else 'valid'
            data_type = 'clean_filename' if 'clean' in wav_file else 'noise_filename'
            info[data_type][key] = wav_file
            info['phase'][key] = phase
            info['speaker'][key] = key[:4]
            info['script_id'][key] = key[-3:]

        # match txt info
        for txt_file in tqdm(txt_file_list):
            key = os.path.basename(txt_file).replace('.txt', '')
            info['text'][key] = txt_file

        print('Matching is completed ...')

        # change meta obj
        self._meta = pd.DataFrame(info)

        # make speaker as indices
        speaker_mappings = {spk: idx for idx, spk in enumerate(sorted(self._meta['speaker'].values))}
        # update infos
        self._meta['speaker'] = [speaker_mappings[spk] for spk in self._meta['speaker'].values]
        self._meta['pass'] = [True] * len(self._meta)

        # read duration
        print('Check durations on wave files ...')
        dur_list = self._process_duration(self._meta['noise_filename'].values, min_wav_rate, max_wav_rate)
        self._meta['duration'] = dur_list

        # text process
        print('Text pre-process ... ')
        self._process_txt(self._meta['text'].values, dur_list, min_txt_rate)

        # filter passed rows
        self._meta = self._meta[self._meta['pass'].values]

        # split train / val
        print('Make train / val meta')
        train_meta = self._meta.query('phase == \'train\'')
        val_meta = self._meta.query('phase != \'train\'')

        # save data frames
        print('Save meta frames on {}'.format(' '.join(self.frame_file_names)))
        self.save_meta(self.frame_file_names, self.meta_path, self._meta, train_meta, val_meta)


def get_datasets(meta_dir: str, batch_size: int, num_workers: int,
                 fix_len: int = 0, skip_audio: bool = False,
                 audio_mask: bool = False) -> Tuple[SpeechDataLoader, SpeechDataLoader]:

    assert os.path.isdir(meta_dir), '{} is not valid directory path!'

    train_file, valid_file = VoiceBankMeta.frame_file_names[1:]

    # load meta file
    train_meta = VoiceBankMeta(os.path.join(meta_dir, train_file))
    valid_meta = VoiceBankMeta(os.path.join(meta_dir, valid_file))

    # create dataset
    train_dataset = SpeechDataset(train_meta, fix_len=fix_len, skip_audio=skip_audio, audio_mask=audio_mask)
    valid_dataset = SpeechDataset(valid_meta, fix_len=fix_len, skip_audio=skip_audio, audio_mask=audio_mask)

    # create data loader
    train_loader = SpeechDataLoader(train_dataset, batch_size=batch_size,
                                    num_workers=num_workers, skip_last_bucket=False)
    valid_loader = SpeechDataLoader(valid_dataset, batch_size=batch_size,
                                    num_workers=num_workers, skip_last_bucket=False)

    return train_loader, valid_loader
