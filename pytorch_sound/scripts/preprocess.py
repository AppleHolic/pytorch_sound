import glob
import os
import fire
from typing import Tuple
from ffmpeg_normalize import FFmpegNormalize
from tqdm import tqdm
from pytorch_sound import settings
from pytorch_sound.data.meta.libri_tts import LibriTTSMeta
from pytorch_sound.data.meta.vctk import VCTKMeta
from pytorch_sound.data.meta.voice_bank import VoiceBankMeta
from pytorch_sound.scripts.libri_tts.fetch import fetch_structure
from pytorch_sound.utils.commons import go_multiprocess


def process_all(args: Tuple[str]):
    in_file, out_file = args

    norm = FFmpegNormalize(normalization_type='rms', audio_codec='pcm_f32le', sample_rate=settings.SAMPLE_RATE)
    norm.add_media_file(in_file, out_file)
    norm.run_normalization()


def read_and_write(args: Tuple[str]):
    in_file, out_file = args
    with open(in_file, 'r') as r:
        with open(out_file, 'w') as w:
            w.write(r.read())


def get_sub_paths(in_dir: str, file_path: str):
    in_dir_abs = os.path.abspath(in_dir)
    sub_ = file_path.replace(in_dir_abs, '')
    if sub_.startswith('/'):
        sub_ = sub_[1:]
    return sub_


def get_sub_dir(in_dir: str, file_path: str):
    sub_path = get_sub_paths(in_dir, file_path)
    sub_path = '/'.join(sub_path.split('/')[:-1])
    return sub_path


class Processor:

    @staticmethod
    def copy_txt(in_dir: str, out_dir: str):
        # make txt list
        print('Lookup file list...')
        in_txt_list = glob.glob(os.path.join(in_dir, '**', '*.txt'))
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
    def __preprocess_wave(in_dir: str, out_dir: str):

        # loop up wave files
        print('Lookup file list...')
        in_wav_list = glob.glob(os.path.join(in_dir, '**', '*.wav'))
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
    def preprocess(in_dir: str, out_dir: str):
        in_wav_list, out_wav_list = __class__.__preprocess_wave(in_dir, out_dir)

        # do multi process
        go_multiprocess(process_all, list(zip(in_wav_list, out_wav_list)))

    @staticmethod
    def voice_bank(in_dir: str, out_dir: str, min_wav_rate: int = 0, max_wav_rate: int = 9999):
        # preprocess audios
        print('Start to process audio files!')
        __class__.preprocess(in_dir, out_dir)

        print('Finishing...')

        # copy texts
        print('Copy text files...')
        __class__.copy_txt(in_dir, out_dir)

        # make meta files
        meta_dir = os.path.join(out_dir, 'meta')
        meta = VoiceBankMeta(meta_dir)
        meta.make_meta(out_dir, min_wav_rate, max_wav_rate, 0)
        print('All processes are finished.')

    @staticmethod
    def libri_tts(in_dir: str, out_dir: str, target_txt: str = 'normalized', is_clean: bool = False):
        # re-construct & copy raw data
        fetch_structure(in_dir, in_dir, target_txt=target_txt, is_clean=is_clean)

        # fetched data dir
        in_dir = os.path.join(in_dir, 'train')

        # preprocess audios
        __class__.preprocess(in_dir, out_dir)

        # copy texts
        __class__.copy_txt(in_dir, out_dir)

        # make meta files
        meta_dir = os.path.join(out_dir, 'meta')
        meta = LibriTTSMeta(meta_dir)
        meta.make_meta(out_dir, settings.MIN_WAV_RATE, settings.MAX_WAV_RATE, settings.MIN_TXT_RATE)

    @staticmethod
    def vctk(in_dir: str, out_dir: str):
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
        go_multiprocess(process_all, list(zip(wave_file_list, out_wav_list)))

        # copy text files
        go_multiprocess(read_and_write, list(zip(txt_file_list, out_txt_list)))

        # make meta files
        meta_dir = os.path.join(out_dir, 'meta')
        meta = VCTKMeta(meta_dir)
        meta.make_meta(out_dir, out_wav_list, out_txt_list)


if __name__ == '__main__':
    fire.Fire(Processor)
