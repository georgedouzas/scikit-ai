# Image components

The components of `skai.image` take images as input, in any form listed in [Media inputs and outputs](media.md).

## Describe images

`ImageDescriptor` returns a `'brief'` or `'detailed'` description of every image, steered by optional `instructions`.
Because it also implements `transform`, its descriptions can feed a text component.

```python exec="true" source="above" result="text"
from PIL import Image
from skai.image import ImageDescriptor

descriptor = ImageDescriptor(detail='detailed', instructions='Mention the colors.', backend='dummy').fit()
print(descriptor.predict([Image.new('RGB', (320, 240), 'navy')]))
```

## Classify images

`ImageClassifier` assigns labels to images zero-shot from `labels`, or few-shot from training images and targets.
Targets outside `labels` raise `ValueError`.

```python exec="true" source="above" result="text"
from PIL import Image
from skai.image import ImageClassifier

images = [Image.new('RGB', (32, 32), color) for color in ('orange', 'gray')]
classifier = ImageClassifier(labels=['cat', 'dog'], backend='dummy').fit()
print(classifier.predict(images), classifier.predict_proba(images).shape)
```

The API reference of every component is in [`skai.image`](../api/image/index.md).
