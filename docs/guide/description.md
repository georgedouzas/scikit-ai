# Description

`skai.description` holds the components that describe media in natural language. `ImageDescriptor` returns a
`'brief'` or `'detailed'` description of every image, steered by optional `instructions`. Because it also implements
`transform`, its descriptions can feed a text component.

```python exec="true" source="above" result="text"
from PIL import Image
from skai.description import ImageDescriptor

descriptor = ImageDescriptor(detail='detailed', instructions='Mention the colors.', backend='dummy').fit()
print(descriptor.predict([Image.new('RGB', (320, 240), 'navy')]))
```

## What fit learns

`fit(X, y)` with reference descriptions as `y` passes image and description pairs to the backend, so new descriptions
follow their style, length and vocabulary.

```python exec="true" source="above" result="text"
from PIL import Image
from skai.description import ImageDescriptor

references = ['Object: square. Color: red.']
descriptor = ImageDescriptor(backend='dummy').fit([Image.new('RGB', (8, 8), 'red')], references)
print(descriptor.examples_[0][1])
```

The API reference is in [`skai.description`](../api/description/index.md).
