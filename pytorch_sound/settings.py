import multiprocessing
from pytorch_sound.data.eng_handler.symbols import symbols as eng_symbols
from pytorch_sound.data.korean import PHONEMES as kor_phonemes, GRAPHEMES as kor_graphemes
from typing import List, Dict

#
# AUDIO, STFT parameters
#
SAMPLE_RATE: int = 22050  # sample rate of target wave
N_FFT: int = 1024
WIN_LENGTH: int = 1024  # STFT window length
HOP_LENGTH: int = 256  # STFT hop length
HOP_STRIDE: int = WIN_LENGTH // HOP_LENGTH  # frames per window
SPEC_SIZE: int = WIN_LENGTH // 2 + 1  # spectrogram bands
MEL_SIZE: int = 80  # mel-spectrogram bands
MFCC_SIZE: int = 40
MEL_MIN: int = 0  # mel minimum freq.
MEL_MAX: int = 8000  # mel maximum freq.
MIN_DB: int = -50  # minimum decibel
MAX_DB: int = 30  # maximum decibel
VN_DB: float = -11.5  # volume normalization target decibel
MULAW_BINS: int = 256  # mu-law quantization bin counts


# Default Preprocess Options
MIN_WAV_RATE: int = 2  # * sample_rate
MAX_WAV_RATE: int = 15
MIN_TXT_RATE: float = 0


# number of workers
NUM_WORKERS: int = multiprocessing.cpu_count() // 2


# english vocabulary
IDX_TO_ENG: List[str] = eng_symbols
ENG_TO_IDX: Dict[str, int] = {x: i + 1 for i, x in enumerate(IDX_TO_ENG[1:])}
ENG_VOCA_SIZE: int = len(IDX_TO_ENG)


# korean vocabularies
IDX_TO_KOR_PHN: List[str] = kor_phonemes
KOR_PHN_TO_IDX: Dict[str, int] = {x: i + 1 for i, x in enumerate(IDX_TO_KOR_PHN[1:])}
KOR_PHN_SIZE: int = len(IDX_TO_KOR_PHN)


IDX_TO_KOR_GRP: List[str] = kor_graphemes
KOR_GRP_TO_IDX: Dict[str, int] = {x: i + 1 for i, x in enumerate(IDX_TO_KOR_GRP[1:])}
KOR_GRP_SIZE: int = len(IDX_TO_KOR_GRP)
