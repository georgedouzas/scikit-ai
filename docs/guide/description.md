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

The API reference is in [`skai.description`](../api/description/index.md).
