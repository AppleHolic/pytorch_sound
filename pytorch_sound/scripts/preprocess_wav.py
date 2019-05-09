import glob
import os
import fire
import librosa
from typing import Any
from scipy.io.wavfile import read as wav_read
from pytorch_sound.utils.commons import go_multiprocess
from pytorch_sound.utils.calculate import volume_norm_log


def load_preproc_write(args: Any):

    in_file, out_file, out_sr, target_db = args

    # load
    sr, wav = wav_read(in_file)

    # resample
    wav = librosa.core.resample(wav, sr, out_sr)

    # volume normalization
    wav = volume_norm_log(wav, target_db)

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
    def preprocess(in_dir: str, out_dir: str, out_sr: int = 22050, target_db: float = 11.5):
        in_wav_list, out_wav_list = __class__.__preprocess_wave(in_dir, out_dir)

        # make args
        sr_list = [int(out_sr)] * len(in_wav_list)
        db_list = [float(target_db)] * len(in_wav_list)

        # do multi process
        go_multiprocess(load_preproc_write, list(zip(in_wav_list, out_wav_list, sr_list, db_list)))


if __name__ == '__main__':
    fire.Fire(Processor)
