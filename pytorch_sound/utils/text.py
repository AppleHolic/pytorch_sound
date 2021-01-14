import unicodedata

import numpy as np
from typing import List

from pytorch_sound.data.eng_handler import text_to_sequence, sequence_to_text
from pytorch_sound import settings


def eng_c2i(sentence: str) -> List[int]:
    """
    Make indices from english to its
    :param sentence: english sentence
    :return: indices of english characters
    """
    # eng to index
    return [settings.ENG_TO_IDX[c] for c in sentence if c in settings.ENG_TO_IDX]


def eng_i2c(idx: int) -> List[str]:
    """
    Make english sentence from indices
    :param idx: indices
    :return: english characters
    """
    # index to eng
    return [settings.IDX_TO_ENG[i] for i in idx if 0 < i < settings.IDX_TO_ENG]


def eng_t2i(txt: str) -> np.array:
    """
    Combined function to make indices from raw english text
    :param txt: english text
    :return: padded indices
    """
    return pad_eng_eos(text_to_sequence(txt, ['english_cleaners']))


def eng_i2t(ixs: List[int]) -> str:
    """
    Make english text from given indices
    :param ixs: indices
    :return: english text
    """
    return sequence_to_text(ixs)


def pad_eng_eos(x: np.array) -> np.array:
    """
    Pad to end of the given indices array.
    :param x: indices array
    :return: padded array
    """
    return np.array(list(x) + [settings.ENG_VOCA_SIZE])


def eng_c2i(sentence: str) -> List[int]:
    """
    Make indices from english to its
    :param sentence: english sentence
    :return: indices of english characters
    """
    # eng to index
    return [settings.ENG_TO_IDX[c] for c in sentence if c in settings.ENG_TO_IDX]


def kor_p2i(phonemes: str) -> List[int]:
    """
    Make indices from korean to its
    :param sentence: korean phonemes
    :return: indices of korean characters
    """
    # eng to index
    return [settings.KOR_PHN_TO_IDX[p] for p in phonemes.split() if p in settings.KOR_PHN_TO_IDX]


def kor_i2p(idx: List[int]) -> List[str]:
    """
    Make korean phonemes from indices
    :param idx: indices
    :return: korean characters
    """
    # index to eng
    return [settings.IDX_TO_KOR_PHN[i] for i in idx if i < len(settings.KOR_PHN_TO_IDX)]


def pad_korp_eos(x: np.array) -> np.array:
    """
    Pad to end of the given indices array.
    :param x: indices array
    :return: padded array
    """
    return np.array(list(x) + [settings.KOR_PHN_SIZE])


def kor_g2i(graphemes: str) -> List[int]:
    """
    Make indices from korean to its
    :param sentence: korean graphemes
    :return: indices of korean characters
    """
    return [settings.KOR_GRP_TO_IDX[p] for p in graphemes if p in settings.KOR_GRP_TO_IDX]


def kor_i2g(idx: List[int]) -> List[str]:
    """
    Make korean graphemes from indices
    :param idx: indices
    :return: korean characters
    """
    # index to eng
    return [settings.IDX_TO_KOR_GRP[i] for i in idx if i < len(settings.IDX_TO_KOR_GRP)]


def pad_korg_eos(x: np.array) -> np.array:
    """
    Pad to end of the given indices array.
    :param x: indices array
    :return: padded array
    """
    return np.array(list(x) + [settings.KOR_GRP_SIZE])


def kor_text2grp(text: str) -> List[str]:
    return unicodedata.normalize('NFD', text)


def kor_grp2text(grp: str) -> List[str]:
    return unicodedata.normalize('NFC', grp)
