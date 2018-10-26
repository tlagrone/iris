

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd


def read_raw_target() -> pd.Series:
    data = np.fromfile('Data/raw/target.csv', dtype=np.dtype(int), sep=',')
    names = np.fromfile('Data/raw/target_names.csv', dtype=np.dtype(str), sep='\n')

    target = pd.Categorical(data)
    target = target.rename_categories({code: name for code, name in enumerate(names)})
    target = pd.Series(target, name='species')
    return target


def read_raw_features() -> pd.DataFrame:
    data = np.fromfile('Data/raw/data.csv', dtype=np.dtype(int), sep=',')
    names = np.fromfile('Data/raw/feature_names.txt', dtype=np.dtype(str), sep='\n')

    features = data.reshape((-1, len(names)))
    features = pd.DataFrame(data, columns=names)
    return features
