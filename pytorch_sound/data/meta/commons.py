import multiprocessing
import re
import random
from collections import defaultdict

from pytorch_sound.utils.sound import get_wav_header


def go_multiprocess(worker_func, inputs):

    # declare pool
    cpu_count = multiprocessing.cpu_count() // 2
    pool = multiprocessing.Pool(cpu_count)

    res = []
    try:
        for i in range(0, len(inputs), cpu_count):
            start_idx, end_idx = i, i + cpu_count
            res += pool.map(worker_func, inputs[start_idx:end_idx])
            print('{}/{}\t{}() processed.'.format(i + 1, len(inputs), worker_func.__name__))
    finally:
        pool.close()

    return res


def get_wav_duration(args):
    speaker, file = args
    try:
        dur = get_wav_header(file)['Duration']
    except:
        dur = -1
    return speaker, file, dur


def get_text_len(args):
    speaker, file, dur = args
    try:
        with open(file, encoding='utf-8') as f:
            # load file
            txt = f.read()
            # merge white spaces into single space
            txt_dur = len(' '.join(txt.split()))
    except:
        txt_dur = -1

    return speaker, file, dur, txt_dur


def clean_eng(args):
    speaker, file, dur = args
    regex = re.compile(r'[a-zA-Z\'\.\,\?\!\ ]+')
    try:
        with open(file, encoding='utf-8') as f:
            # load file
            txt = f.read()
            txt = ' '.join(map(lambda x: x.strip(), regex.findall(txt)))
            # merge white spaces into single space
            txt_dur = len(' '.join(txt.split()))
    except:
        txt_dur = -1

    return speaker, file, dur, txt_dur


def split_train_val_frame(data_frame, val_rate=0.1):
    # total length
    total_len = len(data_frame)

    # split
    idx_list = list(range(total_len))

    if 'speaker' in data_frame:
        temp = defaultdict(list)
        for idx, spk in enumerate(data_frame['speaker'].values):
            temp[spk].append(idx)

        # shuffle
        for key in temp.keys():
            random.shuffle(temp[key])

        train_idx = []
        val_idx = []
        for key in temp.keys():
            split_idx = int(len(temp[key]) * val_rate)
            train_idx.extend(temp[key][split_idx:])
            val_idx.extend(temp[key][:split_idx])
    else:
        # shuffle
        random.shuffle(idx_list)

        # make idx list
        split_idx = int(total_len * val_rate)
        train_idx = idx_list[split_idx:]
        val_idx = idx_list[:split_idx]

    # split data frame
    train_frame = data_frame.iloc[train_idx]
    val_frame = data_frame.iloc[val_idx]

    return train_frame, val_frame
