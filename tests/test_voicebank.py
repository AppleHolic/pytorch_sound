import fire
from tqdm import tqdm
from pytorch_sound.data.meta.voice_bank import get_datasets


def test_voice_bank(meta_dir: str):
    train_loader, valid_loader = get_datasets(meta_dir, 32, 4, 1)
    print('Loop train datasets')
    for _ in tqdm(train_loader):
        pass

    print('Loop valid datasets')
    for _ in tqdm(valid_loader):
        pass


if __name__ == '__main__':
    fire.Fire(test_voice_bank)
