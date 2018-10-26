

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd


def read_target(fp: Union[str, Path]) -> pd.Series:
    data = np.fromfile(fp, dtype=np.dtype(int), sep=',')
    target = pd.Categorical(data)

    names = np.fromfile('Data/raw/target_names.csv', dtype=np.dtype(str), sep='\n')
    target = target.rename_categories({code: name for code, name in enumerate(names)})

    target = pd.Series(target, name='species')
    return target
