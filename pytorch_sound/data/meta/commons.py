import random
from pandas import DataFrame
from collections import defaultdict
from typing import Tuple


def split_train_val_frame(data_frame: DataFrame, val_rate: float = 0.1,
                          label_key: str = 'speaker') -> Tuple[DataFrame, DataFrame]:
    """
    Split DataFrame into train and validation.
    If it has 'speaker' columns, it splits frame using speaker column dependently.
    Not of that, it splits frame randomly.
    :param data_frame: whole data(meta) frame
    :param val_rate: percentage of validation
    :param label_key: standard to be separated with label dist.
    :return: train and valid data(meta) frame
    """
    # total length
    total_len = len(data_frame)

    # split
    idx_list = list(range(total_len))

    if label_key in data_frame:
        temp = defaultdict(list)
        for idx, spk in enumerate(data_frame[label_key].values):
            temp[spk].append(idx)

        # shuffle
        for key in temp.keys():
            random.shuffle(temp[key])

        train_idx = []
        val_idx = []
        for key in temp.keys():
            split_idx = int(len(temp[key]) * val_rate)
            train_idx.extend(temp[key][split_idx:])
            val_idx.extend(temp[key][:split_idx])
    else:
        # shuffle
        random.shuffle(idx_list)

        # make idx list
        split_idx = int(total_len * val_rate)
        train_idx = idx_list[split_idx:]
        val_idx = idx_list[:split_idx]

    # split data frame
    train_frame = data_frame.iloc[train_idx]
    val_frame = data_frame.iloc[val_idx]

    return train_frame, val_frame
