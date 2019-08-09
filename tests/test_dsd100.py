import fire
from tqdm import tqdm
from pytorch_sound.data.meta.dsd100 import get_datasets


def test_dsd100(meta_dir: str):
    """
    Simple function to test correctly loading vctk dataset.
    :param meta_dir: base directory that has meta files
    """
    train_loader, valid_loader = get_datasets(meta_dir, 4, 4, 1)
    print('Loop train datasets')
    for _ in tqdm(train_loader):
        pass

    print('Loop valid datasets')
    for _ in tqdm(valid_loader):
        pass

    print('All of the dataset is loaded successfully.')


if __name__ == '__main__':
    fire.Fire(test_dsd100)
