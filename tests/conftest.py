"""Configuration for the pytest test suite."""

import numpy as np
from sklearn.datasets import fetch_20newsgroups

TRAIN_DATA = fetch_20newsgroups(shuffle=True, random_state=42)
TEST_DATA = fetch_20newsgroups(shuffle=True, random_state=10, subset='test')
CLASSES = np.unique(TRAIN_DATA.target).tolist()
CLASSES_MAPPING = dict(enumerate(TRAIN_DATA.target_names))
X, y = TRAIN_DATA.data, TRAIN_DATA.target
N_TEST_SAMPLES = 5
X_test, y_test = TEST_DATA.data[:N_TEST_SAMPLES], TEST_DATA.target[:N_TEST_SAMPLES]
K_SHOT_NONE_CLASSES_MISSING_DATA = [
    (None, None, TRAIN_DATA.target),
    (CLASSES, TRAIN_DATA.data, None),
    (CLASSES_MAPPING, TRAIN_DATA.data, None),
    (CLASSES, None, None),
    (CLASSES_MAPPING, None, None),
]
K_SHOT_ZERO_CLASSES_MISSING_DATA = [
    (None, None, y),
    (CLASSES_MAPPING, None, y),
    (CLASSES, X, None),
    (CLASSES_MAPPING, X, None),
    (CLASSES, None, None),
    (CLASSES_MAPPING, None, None),
    (None, X, y),
    (CLASSES_MAPPING, X, y),
]
