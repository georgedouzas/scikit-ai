# Classification

`skai.classification` holds the components that assign labels: `TextClassifier`, `ImageClassifier`,
`AudioClassifier`, `VideoClassifier` and `MultimodalClassifier`. They predict zero-shot from `labels`, or few-shot from
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

## Audio and videos

`AudioClassifier` labels sounds, music or speech, and `VideoClassifier` labels videos from `n_frames` frames sampled
evenly over each one, with the audio track on backends that accept audio.

```python exec="true" source="above" result="text"
import numpy as np
from skai.classification import AudioClassifier, VideoClassifier
from skai.core import Video

audio = [(np.zeros(16000, dtype=np.float32), 16000)]
print(AudioClassifier(labels=['dog bark', 'siren'], backend='dummy').fit().predict(audio))
video = Video(np.zeros((20, 8, 8, 3), dtype=np.uint8), fps=10)
print(VideoClassifier(labels=['cooking', 'sports'], n_frames=4, backend='dummy').fit().predict([video]))
```

## Multimodal samples

`MultimodalClassifier` labels samples made of several named parts, such as a photo and its caption, and sends every
part to the backend in one request, each named by its key. Samples are dicts or the rows of a DataFrame.

```python exec="true" source="above" result="text"
from PIL import Image
from skai.classification import MultimodalClassifier

posts = [{'image': Image.new('RGB', (32, 32), 'red'), 'caption': 'Fresh tomatoes for sale.'}]
classifier = MultimodalClassifier(
    labels=['safe', 'unsafe'],
    instructions='Judge the image and the caption together.',
    backend='dummy',
)
print(classifier.fit().predict(posts))
```

## What fit learns

With targets, a classifier learns `classes_` and passes `examples_` to the backend: one example per class by default,
none with `k_shot=0`. With `adaptation='finetune'`, it trains a model on all the training data instead, on backends
that support fine-tuning.

The API reference is in [`skai.classification`](../api/classification/index.md).
