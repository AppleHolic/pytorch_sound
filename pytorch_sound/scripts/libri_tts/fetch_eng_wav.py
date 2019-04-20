import glob
import os
import fire


# Convert LibriTTS Structure
#
# - LibriTTS Structure
# train-clean
# - speaker
# -- book
# --- [id].wav
# --- [id].normalized.txt
# --- [id].original.txt
# train-other
# dev-clean
# dev-other
# test-clean
# TODO: Contain resampling / volume normalization process
def run_detach(data_dir: str, out_dir: str, target_txt: str = 'normalized', is_clean=True):
    # define target data
    if is_clean:
        target_dirs = ['train-clean-360', 'dev-clean']
    else:
        target_dirs = ['train-clean-360', 'train-other-500', 'dev-clean', 'dev-other']
    # chk and make out dir
    out_txt_dir, out_wav_dir = os.path.join(out_dir, '{}', '{}', 'txt'), os.path.join(out_dir, '{}', '{}', 'wav')

    assert target_txt in ['normalized', 'original'], 'target_txt must be "normalized" or "original" !'

    # loop
    for target_name in target_dirs:
        # make path
        target_dir = os.path.join(data_dir, target_name)

        # rename
        mid_dir = 'train' if 'train' in target_name else 'valid'
        speakers = os.listdir(target_dir)
        for spk_id, speaker in enumerate(speakers):
            # make lookup str
            wav_lookup_str = os.path.join(target_dir, speaker, '**', '*.wav')
            txt_lookup_str = os.path.join(target_dir, speaker, '**', '*.{}.txt'.format(target_txt))

            # make cmd
            sub_wav_dir = out_wav_dir.format(mid_dir, speaker)
            sub_txt_dir = out_txt_dir.format(mid_dir, speaker)
            # make output dir
            os.makedirs(sub_wav_dir, exist_ok=True)
            os.makedirs(sub_txt_dir, exist_ok=True)

            # cmd
            wav_copy_cmd = 'cp {} {}'.format(wav_lookup_str, sub_wav_dir)
            txt_copy_cmd = 'cp {} {}'.format(txt_lookup_str, sub_txt_dir)
            conc_cmd = '{} & {}'.format(wav_copy_cmd, txt_copy_cmd)

            # copy
            print('Launch {} !'.format(conc_cmd))
            if spk_id % 10 == 0:
                os.system(conc_cmd)
            else:
                os.system('{} &'.format(conc_cmd))

        # rename txt
        for speaker in speakers:
            sub_txt_dir = out_txt_dir.format(mid_dir, speaker)
            txt_lookup_str = os.path.join(sub_txt_dir, '*.txt')
            for txt_file_path in glob.glob(txt_lookup_str):
                os.rename(txt_file_path, txt_file_path.replace('.{}'.format(target_txt), ''))


if __name__ == '__main__':
    fire.Fire(run_detach)
