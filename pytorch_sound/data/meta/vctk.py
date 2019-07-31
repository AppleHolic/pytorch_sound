import glob

import pandas as pd
import os
from tqdm import tqdm
from typing import List, Tuple, Callable
from pytorch_sound import settings
from pytorch_sound.data.dataset import SpeechDataLoader, SpeechDataset
from pytorch_sound.data.meta import MetaFrame, MetaType
from pytorch_sound.data.meta.commons import split_train_val_frame
from pytorch_sound.utils.sound import get_f0


class VCTKMeta(MetaFrame):
    """
    Extended MetaFrame for using VCTK
    - dataset : https://datashare.is.ed.ac.uk/handle/10283/2651
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

    def make_meta(self, root_dir: str, wav_file_list: List[str] = None, txt_file_list: List[str] = None):
        # composed by three keys
        # train or valid
        # ㄴ id
        # ㄴ ㄴ data type

        # lookup files
        print('Lookup if not provided lists')
        if not wav_file_list:
            wav_file_list = glob.glob(os.path.join(root_dir, '*', 'wav', '*.wav'))
        if not txt_file_list:
            txt_file_list = glob.glob(os.path.join(root_dir, '*', 'txt', '*.txt'))

        infos = {'speaker': {}, 'audio_filename': {}, 'text': {}}

        # match
        wav_match_dict = {os.path.basename(wav_file_path).replace('.wav', ''): wav_file_path
                          for wav_file_path in wav_file_list}
        txt_match_dict = {os.path.basename(txt_file_path).replace('.txt', ''): txt_file_path
                          for txt_file_path in txt_file_list}

        print('Mapping information with their keys')
        for key, wav_file_path in tqdm(wav_match_dict.items()):
            if key in txt_match_dict:
                speaker = wav_file_path.split('/')[-3]
                infos['speaker'][key] = speaker
                infos['audio_filename'][key] = wav_file_path
                infos['text'][key] = txt_match_dict[key]

        # to numeric index
        map_spk = {spk: idx for idx, spk in enumerate(list(set(infos['speaker'].values())))}

        infos['speaker'] = {key: map_spk[val] for key, val in infos['speaker'].items()}

        print('Matching is completed ...')

        # change meta obj
        self._meta = pd.DataFrame(infos)

        # make speaker as indices
        speaker_mappings = {spk: idx for idx, spk in enumerate(sorted(self._meta['speaker'].values))}
        # update infos
        self._meta['speaker'] = [speaker_mappings[spk] for spk in self._meta['speaker'].values]
        self._meta['pass'] = [True] * len(self._meta)

        # read duration
        print('Check durations on wave files ...')
        dur_list = self._process_duration(self._meta['audio_filename'].values,
                                          settings.MIN_WAV_RATE, settings.MAX_WAV_RATE)
        self._meta['duration'] = dur_list

        # text process
        print('Text pre-process ... ')
        self._process_txt(self._meta['text'].values, dur_list, 0.0)

        # filter passed rows
        self._meta = self._meta[self._meta['pass'].values]

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

    train_file, valid_file = VCTKMeta.frame_file_names[1:]

    # load meta file
    train_meta = VCTKMeta(os.path.join(meta_dir, train_file))
    valid_meta = VCTKMeta(os.path.join(meta_dir, valid_file))

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
    meta = VCTKMeta(meta_path)
    meta.make_meta(root_dir)
