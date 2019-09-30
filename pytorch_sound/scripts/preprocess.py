import glob
import os
import fire
import librosa
import numpy as np
from pathlib import Path
from typing import Tuple, List
from ffmpeg_normalize import FFmpegNormalize
from tqdm import tqdm
from pytorch_sound import settings
from pytorch_sound.data.meta.libri_tts import LibriTTSMeta
from pytorch_sound.data.meta.medleydb import MedleyDBMeta
from pytorch_sound.data.meta.vctk import VCTKMeta
from pytorch_sound.data.meta.voice_bank import VoiceBankMeta
from pytorch_sound.data.meta.dsd100 import DSD100Meta
from pytorch_sound.scripts.libri_tts.fetch import fetch_structure
from pytorch_sound.utils.commons import go_multiprocess


def process_all(args: Tuple[str]):
    """
    Audio processing worker function with using ffmpeg.
    Do rms normalization, change codec and sample rate
    :param args: in / out file path
    """
    in_file, out_file, out_sr = args

    norm = FFmpegNormalize(normalization_type='rms', audio_codec='pcm_f32le', sample_rate=out_sr)
    norm.add_media_file(in_file, out_file)
    norm.run_normalization()


def resample(args: Tuple[str]):
    """
    Resampling audio worker function with using ffmpeg.
    Do rms normalization, change codec and sample rate
    :param args: in / out file path
    """
    in_file, out_file, out_sr = args
    command = 'sox {} -ar {} {} rate'.format(in_file, out_sr, out_file)
    os.system(command)


def load_and_numpy_audio(args: Tuple[str]):
    """
    When audio files are very big, it brings more file loading time.
    So, convert audio files to numpy files
    :param args: in / out file path
    """
    in_file, out_file = args

    # load audio file with librosa
    wav, _ = librosa.load(in_file, sr=None)

    # save wav array
    np.save(out_file, wav)


def read_and_write(args: Tuple[str]):
    """
    copy file function
    :param args: in / out file path
    """
    in_file, out_file = args
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


class Processor:

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

        go_multiprocess(read_and_write, list(zip(in_txt_list, out_txt_list)))

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
        go_multiprocess(process_all, list(zip(in_wav_list, out_wav_list, [sample_rate] * len(in_wav_list))))

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
        go_multiprocess(resample, list(zip(in_wav_list, out_wav_list, [sample_rate] * len(in_wav_list))))

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
        go_multiprocess(process_all, list(zip(wave_file_list, out_wav_list, [sample_rate] * len(wave_file_list))))

        # copy text files
        go_multiprocess(read_and_write, list(zip(txt_file_list, out_txt_list)))

        # make meta files
        meta_dir = os.path.join(out_dir, 'meta')
        meta = VCTKMeta(meta_dir)
        meta.make_meta(out_dir, out_wav_list, out_txt_list)

    @staticmethod
    def dsd100(data_dir: str):
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
        go_multiprocess(load_and_numpy_audio, list(zip(mixture_list, out_mixture_list)))
        print('- Vocals File')
        go_multiprocess(load_and_numpy_audio, list(zip(vocals_list, out_vocals_list)))

        meta_dir = os.path.join(data_dir, 'meta')
        meta = DSD100Meta(meta_dir)
        meta.make_meta(data_dir)

    @staticmethod
    def medleydb(in_dir: str):
        """
        preprocess MedleyDB.
        MedleyDB is the dataset has both seperated sound sources and mixture of them.
        Referece Link : https://github.com/marl/medleydb
        :param in_dir: Downloaded path that has V1 and/or V2
        """
        # lookup files
        print('Lookup wave files ...')
        wav_list = list(map(str, Path(in_dir).glob('**/*.wav')))

        # make numpy audio files
        npy_arg_list = [(path, path.replace('.wav', '.npy')) for path in wav_list]

        # run parallel
        print('Save wave files as numpy ...')
        go_multiprocess(load_and_numpy_audio, npy_arg_list)

        meta_dir = os.path.join(in_dir, 'meta')
        meta = MedleyDBMeta(meta_dir)
        meta.make_meta(in_dir)


if __name__ == '__main__':
    fire.Fire(Processor)
