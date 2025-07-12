"""Configuration for the pytest test suite."""

import numpy as np

X = np.array(
    [
        'This is the worst thing I have ever bought.',
        'It does the job, nothing special.',
        'Ia m extremely happy with this purchase!',
        'Very poor quality and bad support.',
        'It is okay, not great but not terrible.',
        'Excellent product! Will buy again.',
        'Disappointed. I expected more.',
        'The item is fine, just as described.',
        'Absolutely fantastic in every way.',
        'Mediocre experience — neither good nor bad.',
    ],
)
y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 1])
X_test = np.array(
    [
        'Superb quality and fast delivery!',
        'Not impressed at all. Would not recommend.',
        'It works as expected, nothing more.',
    ],
)
y_test = np.array([2, 0, 1])
CLASSES = [0, 1, 2]
CLASSES_MAPPING = {
    0: 'negative',
    1: 'neutral',
    2: 'positive',
}
K_SHOT_NONE_CLASSES_MISSING_DATA = [
    (None, None, y),
    (CLASSES, X, None),
    (CLASSES_MAPPING, X, None),
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
