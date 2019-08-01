import os
import pandas as pd
from typing import List, Tuple

from pytorch_sound.data.meta import MetaFrame, MetaType
from pytorch_sound.data.meta.commons import split_train_val_frame
from pytorch_sound.data.dataset import SpeechDataset, SpeechDataLoader


class MaestroMeta(MetaFrame):
    """
    Extended MetaFrame for using Maestro Dataset
    * Especially, the meta files exists in Maestro dataset that can be used on this class
    - dataset : https://magenta.tensorflow.org/datasets/maestro
    """
    def __init__(self, meta_path: str, min_wav_rate: float = 0.0, max_wav_rate: float = 0.0):
        self.meta_path = meta_path
        self.root_dir = os.path.split(self.meta_path)[-2]
        if os.path.exists(self.meta_path):
            if self.meta_path.endswith('csv'):
                self._meta = pd.read_csv(self.meta_path)
            elif self.meta_path.endswith('json'):
                self._meta = pd.read_json(self.meta_path)
            else:
                raise RuntimeError('You should use official maestro meta file !')
        else:
            raise RuntimeError('%s is not exists !'.format(meta_path))
        self.min_wav_rate = min_wav_rate
        self.max_wav_rate = max_wav_rate

    @property
    def columns(self) -> List[Tuple[MetaType, str]]:
        return [(MetaType.AUDIO, 'audio_filename'), (MetaType.MIDI, 'midi_filename'), (MetaType.META, 'duration')]

    @property
    def meta(self) -> pd.DataFrame:
        return self._meta

    @property
    def sr(self) -> int:
        return 44100

    @property
    def frame_file_names(self) -> List[str]:
        return ['maestro-v1.0.0.json', 'maestro-v1.0.0-train.json', 'maestro-v1.0.0-valid.json']

    def __len__(self):
        return len(self._meta)

    def save_meta(self, meta_dir: str, all_frame: pd.DataFrame, train_frame: pd.DataFrame, val_frame: pd.DataFrame):
        # make names
        file_paths = [os.path.join(meta_dir, name) for name in self.frame_file_names]
        # save
        all_frame.to_json(file_paths[0])
        train_frame.to_json(file_paths[1])
        val_frame.to_json(file_paths[2])

    def make_meta(self):
        # convert from related path to absolute path
        audio_file_names = self._meta['audio_filename']
        midi_file_names = self._meta['midi_filename']
        new_audio_files = [os.path.join(self.root_dir, name) for name in audio_file_names]
        new_midi_files = [os.path.join(self.root_dir, name) for name in midi_file_names]
        self._meta['audio_filename'] = new_audio_files
        self._meta['midi_filename'] = new_midi_files

        # split train / val
        print('Make train / val meta')
        train_meta, val_meta = split_train_val_frame(self._meta)

        # save data frames
        print('Save meta frames on {}'.format(' '.join(self.frame_file_names)))
        self.save_meta(self.root_dir, self._meta, train_meta, val_meta)


def get_datasets(meta_dir: str, batch_size: int, num_workers: int,
                 fix_len: int = 0, skip_audio: bool = False) -> Tuple[SpeechDataLoader, SpeechDataLoader]:
    # TODO: update this function in general
    assert os.path.isdir(meta_dir), '{} is not valid directory path!'

    train_file, valid_file = MaestroMeta.frame_file_names[1:]

    # load meta file
    train_meta = MaestroMeta(os.path.join(meta_dir, train_file))
    valid_meta = MaestroMeta(os.path.join(meta_dir, valid_file))

    # create dataset
    train_dataset = SpeechDataset(train_meta, fix_len=fix_len, skip_audio=skip_audio)
    valid_dataset = SpeechDataset(valid_meta, fix_len=fix_len, skip_audio=skip_audio)

    # create data loader
    train_loader = SpeechDataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    valid_loader = SpeechDataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers)

    return train_loader, valid_loader


if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
    meta_path = args[0]
    min_wav_rate, max_wav_rate = list(map(float, args[1:]))
    meta = MaestroMeta(meta_path, min_wav_rate, max_wav_rate)
    meta.make_meta()
