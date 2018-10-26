

import csv
from typing import Tuple

import numpy as np
from sklearn.preprocessing import label_binarize, power_transform


def preprocess(features: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    features = power_transform(features, standardize=True)
    target = label_binarize(target, np.unique(target))
    return features, target


def main():
    features = np.fromfile('Data/raw/data.csv', dtype=np.dtype(int), sep=',')
    target = np.fromfile('Data/raw/target.csv', dtype=np.dtype(int), sep=',')

    features = np.reshape(features, (len(target), -1))

    features, target = preprocess(features, target)

    with open('Data/preprocessed/features.csv', 'w', newline='') as f:
        w = csv.writer(f)
        for row in (features[i,:] for i in range(features.shape[0])):
            w.writerow(row)
    with open('Data/preprocessed/target.csv', 'w', newline='') as f:
        w = csv.writer(f)
        for row in (target[i,:] for i in range(features.shape[0])):
            w.writerow(row)


if __name__ == '__main__':
    main()
