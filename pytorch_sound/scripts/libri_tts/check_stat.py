import fire
import glob
import os
from tqdm import tqdm
from collections import Counter


def print_chr_counter(data_dir: str, log_path: str):
    # info
    info = Counter()
    # lookup str
    print('Start Lookup All Txt Files !')
    all_txt_list = glob.glob(os.path.join(data_dir, '**', '**', 'txt', '*.txt'))
    print('Start count...')
    for txt_file in tqdm(all_txt_list):
        with open(txt_file, 'r') as r:
            for c in r.read():
                info.update(c)

    with open(log_path, 'w') as w:
        w.write(str(info))


if __name__ == '__main__':
    fire.Fire(print_chr_counter)
