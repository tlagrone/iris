import argparse
import os
import sys
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
from sklearn import datasets

RAW_DATA_DIR = os.path.expandvars('$PROJECT_HOME/data/raw')

ROW_SEP = '\n'
COL_SEP = ','
FILE_EXT = 'csv'

BUNCH_TO_FILE = {  # sklearn.Bunch key to base filename
    'data':          f'data.{FILE_EXT}',
    'feature_names': 'feature_names.txt',
    'target':        f'target.{FILE_EXT}',
    'target_names':  f'target_names.{FILE_EXT}',
    'DESCR':         'description.txt'
}

DATA_DTYPE = np.dtype('float')
TARGET_DTYPE = np.dtype('int')
TARGET_NAMES_DTYPE = np.dtype('<U10')


def _list_to_file(list_: list, file: Union[str, Path], *, mode: str='wt') -> None:
    """Write a list to a newline-delimited text file. Do not append a newline."""
    with open(file, mode) as f:
        f.write('\n'.join(list_))


def _list_from_file(file: Union[str, Path]) -> list:
    """Read a list from a newline-delimited text file. Assume no trailing newline."""
    with open(file, 'rt') as f:
        return f.readlines()


def _str_to_file(str_: str, file: Union[str, Path], *, mode: str='wt') -> None:
    with open(file, mode) as f:
        f.write(str_)


# COMBAK Impelement `overwrite_data` body
def write_raw_data(dirout: Union[str, Path]=RAW_DATA_DIR, *,
                   create_dir: bool=True, overwrite_data: bool=False) -> None:
    bunch = datasets.load_iris()

    pathout = Path(dirout)
    if not pathout.exists():
        if create_dir:
            pathout.mkdir()
        else:
            raise FileNotFoundError(pathout)

    paths = {k: pathout.joinpath(v) for k, v in BUNCH_TO_FILE.items()}

    preexisting_files = [fp for fp in paths.values() if fp.exists()]
    if preexisting_files and not overwrite_data:
        raise FileExistsError(preexisting_files)

    bunch['data'].tofile(paths['data'], COL_SEP)
    _list_to_file(bunch['feature_names'], paths['feature_names'])
    bunch['target'].tofile(paths['target'], COL_SEP)
    _list_to_file(bunch['target_names'].tolist(), paths['target_names'])
    _str_to_file(bunch['DESCR'], paths['DESCR'])
