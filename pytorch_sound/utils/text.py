import numpy as np
from typing import List

from pytorch_sound.data.eng_handler import text_to_sequence, sequence_to_text
from pytorch_sound import settings


def eng_c2i(sentence: str) -> List[int]:
    # eng to index
    return [settings.ENG_TO_IDX[c] for c in sentence if c in settings.ENG_TO_IDX]


def eng_i2c(idx: int) -> List[str]:
    # index to eng
    return [settings.IDX_TO_ENG[i] for i in idx if 0 < i < settings.IDX_TO_ENG]


def eng_t2i(txt: str) -> np.array:
    return pad_eng_eos(text_to_sequence(txt, ['english_cleaners']))


def eng_i2t(ixs: List[int]) -> str:
    return sequence_to_text(ixs)


def pad_eng_eos(x: np.array) -> np.array:
    return np.array(list(x) + [settings.ENG_VOCA_SIZE])
