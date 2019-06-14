import glob
import os
import fire
from typing import Any

from pytorch_sound import settings
from pytorch_sound.data.meta.libri_tts import LibriTTSMeta
from pytorch_sound.scripts.libri_tts.fetch_eng_wav import fetch_structure
from pytorch_sound.utils.commons import go_multiprocess


def load_preproc_write(args: Any):
    in_file, out_file, out_sr = args
    cmd = 'sox {} -r {} {} rate '.format(in_file, out_sr, out_file)
    os.system(cmd)


def read_and_write(args: Any):
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
        for d in dirs:
            os.makedirs(os.path.join(out_dir, d), exist_ok=True)

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
        for d in dirs:
            os.makedirs(os.path.join(out_dir, d), exist_ok=True)

        # make out file path list
        out_wav_list = [os.path.join(out_dir, get_sub_paths(in_dir, in_wav_path)) for in_wav_path in in_wav_list]

        return in_wav_list, out_wav_list

    @staticmethod
    def resample(in_dir: str, out_dir: str, out_sr: int = 22050):
        in_wav_list, out_wav_list = __class__.__preprocess_wave(in_dir, out_dir)

        # make args
        sr_list = [int(out_sr)] * len(in_wav_list)

        # do multi process
        go_multiprocess(load_preproc_write, list(zip(in_wav_list, out_wav_list, sr_list)))

    @staticmethod
    def libri_tts(in_dir: str, out_dir: str, out_sr: int = 22050,
                  target_txt: str = 'normalized', is_clean: bool = True):
        # re-construct & copy raw data
        fetch_structure(in_dir, in_dir, target_txt=target_txt, is_clean=is_clean)

        # resample audios
        __class__.resample(in_dir, out_dir, out_sr)

        # copy texts
        __class__.copy_txt(in_dir, out_dir)

        # make meta files
        meta_dir = os.path.join(out_dir, 'meta')
        meta = LibriTTSMeta(meta_dir)
        meta.make_meta(out_dir, settings.MIN_WAV_RATE, settings.MAX_WAV_RATE, settings.MIN_TXT_RATE)


if __name__ == '__main__':
    fire.Fire(Processor)
