import struct
import pretty_midi
from typing import Dict, Any


def parse_midi(path: str):
    midi = None
    try:
        midi = pretty_midi.PrettyMIDI(path)
        midi.remove_invalid_notes()
    except Exception as e:
        raise Exception(("%s\nerror readying midi file %s" % (e, path)))
    return midi


# based on https://blog.theroyweb.com/extracting-wav-file-header-information-using-a-python-script
def get_wav_header(wav_file: str) -> Dict[str, Any]:
    """ Extracts data in the first 44 bytes in a WAV file and writes it
            out in a human-readable format
    """

    def get_data_chunk_location(file):

        # Locate & read data chunk
        data_chunk_location = 0
        file.seek(0, 2)  # Seek to end of file
        input_file_size = file.tell()
        next_chunk_location = 12  # skip the RIFF header
        while True:
            # Read sub chunk header
            file.seek(next_chunk_location)
            buf_hdr = file.read(8)
            if buf_hdr[0:4] == b"data":
                data_chunk_location = next_chunk_location

            next_chunk_location += (8 + struct.unpack('<L', buf_hdr[4:8])[0])
            if next_chunk_location >= input_file_size:
                break

        return data_chunk_location

    # Open file
    with open(wav_file, 'rb') as f:

        # Read in all data
        buf_hdr = f.read(38)

        # Verify that the correct identifiers are present
        if (buf_hdr[0:4] != b"RIFF") or \
                (buf_hdr[12:16] != b"fmt "):
            raise Exception("Input file not a standard WAV file")

        # Wave file header info
        wav_hdr = dict({'ChunkSize': 0, 'Format': '',
                        'Subchunk1Size': 0, 'AudioFormat': 0,
                        'NumChannels': 0, 'SampleRate': 0,
                        'ByteRate': 0, 'BlockAlign': 0,
                        'BitsPerSample': 0, 'Filename': ''})

        # Parse fields
        wav_hdr['ChunkSize'] = struct.unpack('<L', buf_hdr[4:8])[0]
        wav_hdr['Format'] = buf_hdr[8:12]
        wav_hdr['Subchunk1Size'] = struct.unpack('<L', buf_hdr[16:20])[0]
        wav_hdr['AudioFormat'] = struct.unpack('<H', buf_hdr[20:22])[0]
        wav_hdr['NumChannels'] = struct.unpack('<H', buf_hdr[22:24])[0]
        wav_hdr['SampleRate'] = struct.unpack('<L', buf_hdr[24:28])[0]
        wav_hdr['ByteRate'] = struct.unpack('<L', buf_hdr[28:32])[0]
        wav_hdr['BlockAlign'] = struct.unpack('<H', buf_hdr[32:34])[0]
        wav_hdr['BitsPerSample'] = struct.unpack('<H', buf_hdr[34:36])[0]

        # Read data length
        loc = get_data_chunk_location(f)
        if loc > 0:
            f.seek(loc)
            buf_hdr = f.read(8)
            wav_hdr['DataLength'] = struct.unpack('<L', buf_hdr[4:8])[0]
            wav_hdr['Duration'] = wav_hdr['DataLength'] / wav_hdr['SampleRate'] \
                                  / wav_hdr['NumChannels'] / wav_hdr['BitsPerSample'] * 8
        else:
            raise Exception('No data chunk location')

        return wav_hdr
