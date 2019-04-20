import glob
import os
import fire
import librosa
from typing import Any
from pytorch_sound.data.preprocess.commons import go_multiprocess


def load_resample_write(args: Any):

    in_file, out_file, out_sr = args

    # load
    wav, sr = librosa.load(in_file, sr=None)

    # resample
    wav = librosa.core.resample(wav, sr, out_sr)

    # write
    librosa.output.write_wav(out_file, wav, out_sr)


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


class Resampler:

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
    def resample(in_dir: str, out_dir: str, out_sr: str):

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

        # do multi process
        go_multiprocess(load_resample_write, list(zip(in_wav_list, out_wav_list, [int(out_sr)] * len(in_wav_list))))


if __name__ == '__main__':
    fire.Fire(Resampler)
