import csv
import numpy as np
from typing import List, Tuple

from pytorch_sound.data.eng_handler import text_to_sequence, sequence_to_text
from pytorch_sound.utils.settings import CFG


def read_tsv(in_path: str) -> List[Tuple]:
    with open(in_path, encoding='utf-8') as f:
        return list(csv.reader(f, delimiter='\t'))


def save_tsv(out_path: str, rows: int):
    with open(out_path, 'w', encoding='utf-8') as f:
        csv.writer(f, delimiter='\t', quoting=csv.QUOTE_MINIMAL).writerows(rows)


def eng_c2i(sentence: str) -> List[int]:
    # eng to index
    return [CFG.ENG_TO_IDX[c] for c in sentence if c in CFG.ENG_TO_IDX]


def eng_i2c(idx: int) -> List[str]:
    # index to eng
    return [CFG.IDX_TO_ENG[i] for i in idx if 0 < i < CFG.IDX_TO_ENG]


def eng_t2i(txt: str) -> np.array:
    return pad_eng_eos(text_to_sequence(txt, ['english_cleaners']))


def eng_i2t(ixs: List[int]) -> str:
    return sequence_to_text(ixs)


def pad_eng_eos(x: np.array) -> np.array:
    return np.array(list(x) + [CFG.ENG_VOCA_SIZE])
