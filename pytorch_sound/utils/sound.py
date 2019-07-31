import pretty_midi
import numpy as np
import librosa
import pyworld
from scipy.io.wavfile import read as wav_read
from pysndfx import AudioEffectsChain


def parse_midi(path: str) -> np.ndarray:
    """
    simple parsing function to make piano-roll from midi file
    :param path: the MIDI file path
    :return: an array of piano-roll
    """
    midi = None
    try:
        midi = pretty_midi.PrettyMIDI(path)
        midi.remove_invalid_notes()
    except Exception as e:
        raise Exception(("%s\nerror readying midi file %s" % (e, path)))
    return midi


def lowpass(wav: np.ndarray, frequency: int) -> np.ndarray:
    """
    adopt lowpass using pysndfx package
    :param wav: wav-form numpy array
    :param frequency: target frequency
    :return: filtered wav
    """
    fx = (
        AudioEffectsChain().lowpass(frequency=frequency)
    )
    return fx(wav)


def get_f0(wav: np.array, hop_length: int, sr: int = 22050):
    """
    Parse f0 feature from given wave with using WORLD Vocoder
    :param wav: an array of wave
    :param hop_length: hop(stride) length
    :param sr: sample rate of wave
    :return: f0 feature
    """
    x = librosa.util.pad_center(wav, len(wav), mode='reflect').astype('double')
    _f0, t = pyworld.dio(x, sr, frame_period=hop_length / sr * 1e+3)  # raw pitch extractor
    f0 = pyworld.stonemask(x, _f0, t, sr)  # pitch refinement
    return f0.astype(np.float32)


def get_wav_duration(file: str) -> float:
    """
    Calc duration of wave file
    :param file: file path
    :return: wave duration in seconds
    """
    try:
        sr, wav = wav_read(file)
        dur = len(wav) / sr
    except:
        dur = -1
    return dur
