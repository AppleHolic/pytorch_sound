import glob
import os
import fire
import librosa
import numpy as np
import pandas as pd

from collections import defaultdict
from pathlib import Path
from typing import Tuple, List
from ffmpeg_normalize import FFmpegNormalize
from tqdm import tqdm
from joblib import Parallel, delayed, cpu_count
from pytorch_sound import settings
from pytorch_sound.data.meta.libri_tts import LibriTTSMeta
from pytorch_sound.data.meta.ljspeech import LJSpeechMeta
from pytorch_sound.data.meta.medleydb import MedleyDBMeta
from pytorch_sound.data.meta.vctk import VCTKMeta
from pytorch_sound.data.meta.voice_bank import VoiceBankMeta
from pytorch_sound.data.meta.dsd100 import DSD100Meta
from pytorch_sound.scripts.libri_tts.fetch import fetch_structure


def process_all(in_file: str, out_file: str, out_sr: int):
    """
    Audio processing worker function with using ffmpeg.
    Do rms normalization, change codec and sample rate
    """
    norm = FFmpegNormalize(normalization_type='rms', audio_codec='pcm_f32le', sample_rate=out_sr)
    norm.add_media_file(in_file, out_file)
    norm.run_normalization()


def load_split_numpy(in_file: str, out_file: str, wav_len: int):
    """
    When audio files are very big, it brings more file loading time.
    So, convert audio files to numpy files
    """
    # load audio file with librosa
    wav, _ = librosa.load(in_file, sr=None)

    # save wav array
    for idx in range(0, len(wav) - wav_len, wav_len):
        np.save(out_file.replace('.npy', '.{}.npy'.format(idx)), wav[idx: idx+wav_len])


def resample(in_file: str, out_file: str, out_sr: int):
    """
    Resampling audio worker function with using ffmpeg.
    Do rms normalization, change codec and sample rate
    """
    command = 'sox {} -ar {} {} rate'.format(in_file, out_sr, out_file)
    os.system(command)


def load_and_numpy_audio(in_file: str, out_file: str):
    """
    When audio files are very big, it brings more file loading time.
    So, convert audio files to numpy files
    """
    try:
        # load audio file with librosa
        wav, _ = librosa.load(in_file, sr=None)

        if len(wav.shape) > 1:
            wav = wav[0]

        # save wav array
        np.save(out_file, wav)
    except Exception:
        print('Failed to convert on {}'.format(str((in_file, out_file))))


def read_and_write(in_file: str, out_file: str):
    """
    copy file function
    :param args: in / out file path
    """
    with open(in_file, 'r') as r:
        with open(out_file, 'w') as w:
            w.write(r.read())


def get_sub_paths(in_dir: str, file_path: str):
    """
    :param in_dir: base directory of data files
    :param file_path: target file path
    :return: parsed sub paths
    """
    in_dir_abs = os.path.abspath(in_dir)
    sub_ = file_path.replace(in_dir_abs, '')
    if sub_.startswith('/'):
        sub_ = sub_[1:]
    return sub_


def get_sub_dir(in_dir: str, file_path: str):
    """
    :param in_dir: base directory of data files
    :param file_path: target file path
    :return: parsed sub directory path
    """
    sub_path = get_sub_paths(in_dir, file_path)
    sub_path = '/'.join(sub_path.split('/')[:-1])
    return sub_path


def partialize_npy_wave(npy_path: str, num_partial_sample: int, min_partial_length: int = 0):
    # load numpy wave
    arr = np.load(npy_path)
    if len(arr.shape) > 1:
        arr = arr[0]
    num_partial_sample = int(num_partial_sample)

    # loop
    partial_path_list = []
    for idx, start_idx in enumerate(range(0, len(arr), num_partial_sample)):
        # partial output path
        partial_out_path = npy_path.replace('.npy', '_{:05d}.npy'.format(idx + 1))

        end_idx = start_idx + num_partial_sample
        partial_arr = arr[start_idx:end_idx]

        if min_partial_length:
            if len(partial_arr) < min_partial_length:
                continue

        np.save(partial_out_path, partial_arr)
        partial_path_list.append(partial_out_path)

    return npy_path, partial_path_list


class Processor:

    num_workers = cpu_count() // 2

    @staticmethod
    def __copy_txt(in_dir: str, out_dir: str):
        """
        helper function to copy text files recursively
        :param in_dir: base directory of data files
        :param out_dir: target directory
        """
        # make txt list
        print('Lookup file list...')
        in_txt_list = glob.glob(os.path.join(in_dir, '*.txt'))
        in_txt_list += glob.glob(os.path.join(in_dir, '**', '*.txt'))
        in_txt_list += glob.glob(os.path.join(in_dir, '**', '**', '*.txt'))
        in_txt_list += glob.glob(os.path.join(in_dir, '**', '**', '**', '*.txt'))

        dirs = list(map(lambda x: get_sub_dir(in_dir, x), in_txt_list))

        # make dirs
        print('Start to make sub directories...')
        out_dir_list = list(set([os.path.join(out_dir, d) for d in dirs]))
        for d in out_dir_list:
            os.makedirs(d, exist_ok=True)

        out_txt_list = [os.path.join(out_dir, get_sub_paths(in_dir, in_txt_path)) for in_txt_path in in_txt_list]

        Parallel(n_jobs=__class__.num_workers)(
            delayed(read_and_write)(*args) for args in tqdm(zip(in_txt_list, out_txt_list))
        )

    @staticmethod
    def __get_wave_file_list(in_dir: str, out_dir: str) -> Tuple[List[str], List[str]]:
        """
        lookup wave files and match file path on target directory
        :param in_dir: base directory of data files
        :param out_dir: target directory
        :return: base wave file list, target wave file list
        """

        # loop up wave files
        print('Lookup file list...')
        in_wav_list = glob.glob(os.path.join(in_dir, '*.wav'))
        in_wav_list += glob.glob(os.path.join(in_dir, '**', '*.wav'))
        in_wav_list += glob.glob(os.path.join(in_dir, '**', '**', '*.wav'))
        in_wav_list += glob.glob(os.path.join(in_dir, '**', '**', '**', '*.wav'))

        # parse directories
        dirs = list(map(lambda x: get_sub_dir(in_dir, x), in_wav_list))

        # make dirs
        print('Start to make sub directories...')
        out_dir_list = list(set([os.path.join(out_dir, d) for d in dirs]))
        for d in out_dir_list:
            os.makedirs(d, exist_ok=True)

        # make out file path list
        out_wav_list = [os.path.join(out_dir, get_sub_paths(in_dir, in_wav_path)) for in_wav_path in in_wav_list]

        return in_wav_list, out_wav_list

    @staticmethod
    def preprocess_audio(in_dir: str, out_dir: str, sample_rate: int = 22050):
        """
        Preprocess audios given base directory and target directory with multi thread function.
        :param in_dir: base directory of data files
        :param out_dir: target directory
        :param sample_rate: target audio sample rate
        """
        in_wav_list, out_wav_list = __class__.__get_wave_file_list(in_dir, out_dir)

        # do multi process
        Parallel(n_jobs=__class__.num_workers)(
            delayed(read_and_write)
            (*args, sample_rate) for args in tqdm(zip(in_wav_list, out_wav_list))
        )

    @staticmethod
    def resample_audio(in_dir: str, out_dir: str, sample_rate: int):
        """
        Resample audios given base directory and target directory with multi thread function.
        :param in_dir: base directory of data files
        :param out_dir: target directory
        :param sample_rate: target audio sample rate
        """
        in_wav_list, out_wav_list = __class__.__get_wave_file_list(in_dir, out_dir)

        # do multi process
        Parallel(n_jobs=__class__.num_workers)(
            delayed(resample)
            (*args, sample_rate) for args in tqdm(zip(in_wav_list, out_wav_list))
        )

    @staticmethod
    def voice_bank(in_dir: str, out_dir: str, min_wav_rate: int = 0,
                   max_wav_rate: int = 9999, sample_rate: int = 22050):
        """
        Pre-process from downloaded voice bank files to loadable files and make meta files
        :param in_dir: base directory of data files
        :param out_dir: target directory
        :param min_wav_rate: minimum wave duration
        :param max_wav_rate: maximum wave duration
        :param sample_rate: target audio sample rate
        """
        # preprocess audios
        print('Start to process audio files!')
        __class__.preprocess_audio(in_dir, out_dir, sample_rate=sample_rate)

        print('Finishing...')

        # copy texts
        print('Copy text files...')
        __class__.__copy_txt(in_dir, out_dir)

        # make meta files
        meta_dir = os.path.join(out_dir, 'meta')
        meta = VoiceBankMeta(meta_dir)
        meta.make_meta(out_dir, min_wav_rate, max_wav_rate, 0)
        print('All processes are finished.')

    @staticmethod
    def libri_tts(in_dir: str, out_dir: str, target_txt: str = 'normalized', is_clean: bool = False):
        """
        Pre-process from downloaded LibriTTS files to loadable files and make meta files
        :param in_dir: base directory of data files
        :param out_dir: target directory
        :param target_txt: LibriTTS has two types of text, choose one in [normalized, original]
        :param is_clean: LibriTTS has clean dataset and noisy dataset. True is clean and the another.
        """
        # re-construct & copy raw data
        fetch_structure(in_dir, in_dir, target_txt=target_txt, is_clean=is_clean)

        # fetched data dir
        in_dir = os.path.join(in_dir, 'train')

        # preprocess audios
        __class__.preprocess_audio(in_dir, out_dir)

        # copy texts
        __class__.__copy_txt(in_dir, out_dir)

        # make meta files
        meta_dir = os.path.join(out_dir, 'meta')
        meta = LibriTTSMeta(meta_dir)
        meta.make_meta(out_dir, settings.MIN_WAV_RATE, settings.MAX_WAV_RATE, settings.MIN_TXT_RATE)

    @staticmethod
    def vctk(in_dir: str, out_dir: str, sample_rate: int = 22050):
        """
        Pre-process from downloaded VCTK files to loadable files and make meta files.
        It is processed with default audio settings. See pytorch_sound/settings.py
        :param in_dir: base directory of data files
        :param out_dir: target directory
        """
        # lookup files
        print('lookup files...')
        wave_file_list = glob.glob(os.path.join(in_dir, 'wavs', '*', '*.wav'))
        txt_file_list = glob.glob(os.path.join(in_dir, 'txt', '*', '*.txt'))

        # make output file path list
        print('Make out file list...')
        out_wav_list = []
        for wav_file_path in wave_file_list:
            spk, file_name = wav_file_path.split('/')[-2:]
            out_wav_path = os.path.join(out_dir, spk, 'wav', file_name)
            out_wav_list.append(out_wav_path)

        out_txt_list = []
        for txt_file_path in txt_file_list:
            spk, file_name = txt_file_path.split('/')[-2:]
            out_txt_path = os.path.join(out_dir, spk, 'txt', file_name)
            out_txt_list.append(out_txt_path)

        # make directories
        print('Make directories...')
        out_wav_dirs = list(set([os.path.dirname(out_wav_path) for out_wav_path in out_wav_list]))
        out_txt_dirs = list(set([os.path.dirname(out_txt_path) for out_txt_path in out_txt_list]))

        for d in tqdm(out_wav_dirs + out_txt_dirs):
            if not os.path.exists(d):
                os.makedirs(d)

        # preprocess audio files
        print('Start Audio Processing ...')
        Parallel(n_jobs=__class__.num_workers)(
            delayed(process_all)
            (*args, sample_rate) for args in tqdm(zip(wave_file_list, out_wav_list))
        )

        # copy text files
        Parallel(n_jobs=__class__.num_workers)(
            delayed(read_and_write)
            (*args) for args in tqdm(zip(txt_file_list, out_txt_list))
        )

        # make meta files
        meta_dir = os.path.join(out_dir, 'meta')
        meta = VCTKMeta(meta_dir)
        meta.make_meta(out_dir, out_wav_list, out_txt_list)

    @staticmethod
    def ljspeech(in_dir: str, out_dir: str, meta_csv_path: str, sample_rate: int = 22050):
        # lookup files
        print('lookup files ...')
        wav_file_list = glob.glob(os.path.join(in_dir, 'wavs', '*.wav'))
        txt_info = pd.read_csv(meta_csv_path, header=None, sep='|')
        txt_info.columns = ['id', 'text', 'normalized_text']

        # make directory
        os.makedirs(os.path.join(out_dir, 'wavs'), exist_ok=True)

        # make output file list
        out_dir = os.path.abspath(out_dir)
        out_wav_list = [os.path.join(out_dir, 'wavs', os.path.basename(wav_path)) for wav_path in wav_file_list]

        # preprocess audio files
        print('Start Audio Processing ...')
        # Parallel(n_jobs=__class__.num_workers)(
        #     delayed(process_all)
        #     (*args, sample_rate) for args in tqdm(zip(wav_file_list, out_wav_list))
        # )

        # make meta files
        meta_dir = os.path.join(out_dir, 'meta')
        meta = LJSpeechMeta(meta_dir)
        meta.make_meta(out_wav_list, txt_info)

    @staticmethod
    def dsd100(data_dir: str, wav_subset_len: int = 44100 * 10):
        """
        DSD100 is different to others, it just make meta file to load directly original ones.
        :param data_dir: Data root directory
        """
        print('Lookup files ...')
        mixture_list = glob.glob(os.path.join(data_dir, 'Mixtures', '**', '**', 'mixture.wav'))
        out_mixture_list = [file_path.replace('.wav', '.npy') for file_path in mixture_list]

        # It only extract vocals. If you wanna use other source, override it.
        vocals_list = glob.glob(os.path.join(data_dir, 'Sources', '**', '**', 'vocals.wav'))
        out_vocals_list = [file_path.replace('.wav', '.npy') for file_path in vocals_list]

        # save as numpy file
        print('Save as numpy files..')
        print('- Mixture File')
        Parallel(n_jobs=__class__.num_workers)(
            delayed(load_split_numpy)
            (*args, wav_subset_len) for args in tqdm(zip(mixture_list, out_mixture_list))
        )
        print('- Vocals File')
        Parallel(n_jobs=__class__.num_workers)(
            delayed(load_split_numpy)
            (*args, wav_subset_len) for args in tqdm(zip(vocals_list, out_vocals_list))
        )

        meta_dir = os.path.join(data_dir, 'meta')
        meta = DSD100Meta(meta_dir)
        meta.make_meta(data_dir)

    @staticmethod
    def medleydb(in_dir: str, wav_subset_seconds: int = 10):
        """
        preprocess MedleyDB.
        MedleyDB is the dataset has both separated sound sources and mixture of them.
        Referece Link : https://github.com/marl/medleydb
        :param in_dir: Downloaded path that has V1 and/or V2
        """
        # lookup files
        print('Lookup wave files ...')
        wav_list = list(map(str, Path(in_dir).glob('**/*.wav')))

        # make numpy audio files
        # run parallel
        print('Save wave files as numpy ...')
        Parallel(n_jobs=__class__.num_workers)(
            delayed(load_and_numpy_audio)
            (path, path.replace('.wav', '.npy')) for path in tqdm(wav_list)
        )

        meta_dir = os.path.join(in_dir, 'meta')
        meta = MedleyDBMeta(meta_dir)
        meta.make_meta(in_dir)

        print('Start to make partitions!')
        __class__.partialize(meta_dir, wav_subset_seconds)

    @staticmethod
    def partialize(meta_dir: str, seconds: int):
        # load meta inst
        print('Load meta information ...')
        train_file, valid_file = MedleyDBMeta.frame_file_names[1:]
        train_file = os.path.join(meta_dir, train_file)
        valid_file = os.path.join(meta_dir, valid_file)

        # load meta file
        train_meta = MedleyDBMeta(train_file)
        valid_meta = MedleyDBMeta(valid_file)
        sample_length = seconds * train_meta.sr
        target_columns = ['mixture_filename', 'voice_filename']

        # backup
        read_and_write(train_file, train_file.replace('.json', '_backup.json'))
        read_and_write(valid_file, valid_file.replace('.json', '_backup.json'))

        # collect all files
        target_files = []
        for column in target_columns:
            target_files.extend(train_meta.meta[column].values)
            target_files.extend(valid_meta.meta[column].values)

        # do parallel
        print('Partializing process is started.')
        num_workers = cpu_count() // 2

        results = Parallel(n_jobs=num_workers)(
            delayed(partialize_npy_wave)
            (target_file, sample_length, sample_length) for target_file in tqdm(target_files)
        )

        # make new meta
        print('Make new meta, and write it on {}\t{}'.format(train_file, valid_file))

        partial_keymap = {npy_path: partial_path_list for npy_path, partial_path_list in results}

        # train meta
        train_info = defaultdict(list)
        valid_info = defaultdict(list)
        for idx, series in train_meta.meta.iterrows():
            mixture_filename, voice_filename = series['mixture_filename'], series['voice_filename']
            mix_parts = partial_keymap[mixture_filename]
            voice_parts = partial_keymap[voice_filename]

            if len(mix_parts) != len(voice_parts):
                print(str(series))
                continue

            train_info['mixture_filename'].extend(mix_parts)
            train_info['voice_filename'].extend(voice_parts)

        for idx, series in valid_meta.meta.iterrows():
            mixture_filename, voice_filename = series['mixture_filename'], series['voice_filename']
            mix_parts = partial_keymap[mixture_filename]
            voice_parts = partial_keymap[voice_filename]

            if len(mix_parts) != len(voice_parts):
                print(str(series))
                continue

            valid_info['mixture_filename'].extend(mix_parts)
            valid_info['voice_filename'].extend(voice_parts)

        # make data frame
        new_train_df = pd.DataFrame(dict(train_info))
        new_valid_df = pd.DataFrame(dict(valid_info))

        new_train_df.to_json(train_file)
        new_valid_df.to_json(valid_file)


if __name__ == '__main__':
    fire.Fire(Processor)
