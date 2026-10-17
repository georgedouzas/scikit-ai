# Classification

`skai.classification` holds the components that assign labels. They predict zero-shot from `labels`, or few-shot from
training samples and targets, and pass `k_shot` training samples to the backend as examples. `labels` can be a dict
that maps every label to a description, and `instructions` adds guidance. Targets can be labels or a binary indicator
matrix for multilabel classification. Targets outside `labels` raise `ValueError`.

## Texts

```python exec="true" source="above" result="text"
from skai.classification import TextClassifier

zero_shot = TextClassifier(labels={'bug': 'a defect report', 'feature': 'a request for something new'}, backend='dummy')
print(zero_shot.fit().predict(['The app crashes on start.', 'Please add dark mode.']))

few_shot = TextClassifier(k_shot=2, random_state=0, backend='dummy')
few_shot.fit(['Loved it.', 'Hated it.', 'Superb.', 'Awful.'], ['positive', 'negative', 'positive', 'negative'])
print(few_shot.predict_proba(['Not bad at all.']).round(2))
```

## Images

```python exec="true" source="above" result="text"
from PIL import Image
from skai.classification import ImageClassifier

images = [Image.new('RGB', (32, 32), color) for color in ('orange', 'gray')]
classifier = ImageClassifier(labels=['cat', 'dog'], backend='dummy').fit()
print(classifier.predict(images), classifier.predict_proba(images).shape)
```

The API reference is in [`skai.classification`](../api/classification/index.md).
