import os
import yaml
import glob
import pandas as pd
import numpy as np
from typing import List, Tuple
from pytorch_sound.data.dataset import SpeechDataLoader
from pytorch_sound.data.meta import MetaFrame, MetaType


# asset directory
from pytorch_sound.data.meta.commons import split_train_val_frame
from pytorch_sound.utils.commons import go_multiprocess

MEDLEYDB_META_DIR = os.path.abspath(os.path.join(__file__, '..', '..', '..', '..',
                                                 'assets', 'medleydb_metafiles'))


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

    def make_meta(self, root_dir: str, meta_dir: str = None):
        if not meta_dir:
            meta_dir = MEDLEYDB_META_DIR

        # 1. load meta
        print('Load MedleyDB meta info')
        meta = load_meta(meta_dir)

        # 2. filter meta from given audio files
        meta_match_mixkey = {record['mix_filename'].replace('.wav', '.npy'): record for record in meta}

        # 3. load all mixture files
        print('Lookup mix files')
        mix_file_list = glob.glob(os.path.join(root_dir, '**', '**', '*.npy'))

        # 4. get mix/vocal pairs
        print('Matching mix / vocal pairs')
        pair_meta = get_mix_vocal_pairs(mix_file_list, meta_match_mixkey, ext='npy')

        # Each songs have a vocal track or several vocal tracks.
        # So, If the song has several tracks, It can be merged as one file being able to load their own fastly.
        # "pair_meta" looks like, {"mixture_filename": ["first_vocal_filename", "second..."]}

        # Make a function to load and to merge vocal files
        mix_paths, voice_paths = map(list, zip(*pair_meta.items()))

        # do parallel
        print('Merging multi-vocal-tracks ...')
        results = go_multiprocess(load_and_merge_audios, list(pair_meta.items()))
        out_path_list, source_numbers = map(list, zip(*results))

        # make meta values
        filtered_zips = [(m, v, s) for m, v, s in zip(mix_paths, out_path_list, source_numbers) if v != -1]
        mix_results, voice_results, source_numbers = map(list, zip(*filtered_zips))

        # make track length column
        voice_track_lengths = []
        for v, s in zip(voice_results, source_numbers):
            if s < 2:
                voice_track_lengths.append(s)
            else:
                voice_track_lengths.append(2)

        # make meta
        self._meta['mixture_filename'] = mix_results
        self._meta['voice_filename'] = voice_results
        self._meta['voice_tracks'] = voice_track_lengths

        print('Make train / val meta')
        train_meta, val_meta = split_train_val_frame(self._meta, val_rate=0.1, label_key='voice_tracks')

        print('Save meta frames on {}'.format(' '.join(self.frame_file_names)))
        self.save_meta(self.frame_file_names, self.meta_path, self._meta, train_meta, val_meta)

        print('Done')


def load_and_merge_audios(mix_path: str, audio_npy_list: List[str]):
    # make output path
    try:
        if len(audio_npy_list) == 0:
            out_path = mix_path.replace('MIX.npy', '_voice.npy')
            # load mix file
            mix_wav = np.load(mix_path)
            np.save(out_path, np.zeros_like(mix_wav))
        elif len(audio_npy_list) == 1:
            out_path = audio_npy_list[0][:-6] + 'voice.npy'
            os.system('cp {} {}'.format(audio_npy_list[0], out_path))
        else:
            out_path = audio_npy_list[0][:-6] + 'voice.npy'
            # load
            audios = [np.load(npy_path)[np.newaxis, ...] for npy_path in audio_npy_list]

            # concat and return
            audio = np.mean(audios, axis=0)
            np.save(out_path, audio)
    except Exception:
        return -1
    return out_path, len(audio_npy_list)


def load_meta(dir_path: str) -> List[str]:
    # scan files
    file_names = os.listdir(dir_path)

    # load and append meta list
    meta_list = []
    for file_path in file_names:
        meta_path = os.path.join(dir_path, file_path)
        with open(meta_path, 'r') as r:
            meta = yaml.safe_load(r)
        meta_list.append(meta)

    return meta_list


def get_mix_vocal_pairs(mix_file_list: List[str], meta_match_mixkey: List[str], ext: str = None) -> List[str]:
    # scan mix and vocal files
    # key : mix file path , value : list of vocal files
    vocal_files = {}

    # fetch ext
    if ext:
        ext = ext if ext.startswith('.') else '.' + ext

    for mix_file_name in mix_file_list:
        # parse key
        key = os.path.basename(mix_file_name)
        # parse stem directory
        audio_dir = os.path.dirname(mix_file_name)
        stem_dir_name = os.path.basename(audio_dir) + '_STEMS'
        stem_dir = os.path.join(audio_dir, stem_dir_name)
        meta = meta_match_mixkey[key]
        vocal_files[mix_file_name] = []

        for key, val in meta['stems'].items():
            if isinstance(val['instrument'], list):
                for inst in val['instrument']:
                    if 'singer' in inst or 'vocal' in inst:
                        file_path = val['filename'] if not ext else val['filename'].replace('.wav', ext)
                        file_path = os.path.join(stem_dir, file_path)
                        vocal_files[mix_file_name].append(file_path)
                        break
            else:
                if 'singer' in val['instrument'] or 'vocal' in val['instrument']:
                    file_path = val['filename'] if not ext else val['filename'].replace('.wav', ext)
                    file_path = os.path.join(stem_dir, file_path)
                    vocal_files[mix_file_name].append(file_path)

    return vocal_files


def get_datasets(meta_dir: str, batch_size: int, num_workers: int,
                 fix_len: int = 0, audio_mask: bool = False) -> Tuple[SpeechDataLoader, SpeechDataLoader]:
    raise NotImplementedError('To be coded')


if __name__ == '__main__':
    import sys
    input_dir = sys.argv[1]
    meta_dir = os.path.join(input_dir, 'meta')
    meta = MedleyDBMeta(meta_dir)
    meta.make_meta(input_dir)
