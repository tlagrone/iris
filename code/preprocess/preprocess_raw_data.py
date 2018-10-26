

from typing import Tuple

import numpy as np
from sklearn.preprocessing import label_binarize, power_transform


def preprocess(features: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    features = power_transform(features, standardize=True)
    target = label_binarize(target, np.unique(target))
    return features, target
