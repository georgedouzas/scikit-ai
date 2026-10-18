# Extraction

`skai.extraction` holds the components that extract structured records: `TextExtractor`, `ImageExtractor`,
`AudioExtractor`, `VideoExtractor` and `MultimodalExtractor`. Each extracts the required `fields` from every sample,
steered by optional `instructions`. `predict` returns one dict per sample whose keys are exactly the field names, with
`None` for a field the sample does not contain. An extractor fitted without `fields` raises `InvalidParameterError`.

```python exec="true" source="above" result="text"
from skai.extraction import TextExtractor

extractor = TextExtractor(fields={'name': 'person name', 'city': 'city of residence'}, backend='dummy')
print(extractor.fit().predict(['Ada lives in Athens.']))
```

## Nested and list fields

A field is declared in one of three ways:

- A description string, for one value.
- A dict of subfields, for a nested record.
- A one-element list that holds a description or a dict, for a list of values or records.

Extracted lists are Python lists, so a storyboard is a list of scene records that `ForEach` can process scene by
scene, as [Pipelines](pipelines.md) shows.

```python exec="true" source="above" result="text"
from skai.extraction import TextExtractor

fields = {
    'title': 'the title of the video',
    'tags': ['a topic of the brief'],
    'scenes': [{'narration': 'spoken text', 'visual': 'what is shown'}],
}
storyboard = TextExtractor(fields=fields, backend='dummy').fit().predict(['A brief about solar panels.'])[0]
print(storyboard['title'], storyboard['tags'])
print(storyboard['scenes'][0])
```

## Fields as columns

`transform` returns a two-dimensional object array with one column per top-level field, in the order of `fields`, and
`get_feature_names_out` returns the field names. With `set_output(transform='pandas')`, the columns form a DataFrame
that can feed a `ColumnTransformer`. `Field` from `skai.pipeline` picks one field of the records of `predict`.

```python exec="true" source="above" result="text"
from skai.extraction import TextExtractor

extractor = TextExtractor(fields={'name': 'person name', 'city': 'city of residence'}, backend='dummy')
print(extractor.set_output(transform='pandas').fit_transform(['The name is Ada.', 'The city is Athens.']))
```

## Images, audio, videos and multimodal samples

`ImageExtractor` reads fields from images, such as the vendor and total of an invoice scan. `AudioExtractor` reads
them from audio, and `VideoExtractor` from `n_frames` frames sampled evenly over every video, with the audio track on
backends that accept audio. `MultimodalExtractor` sends every named part of a sample in one request.

```python exec="true" source="above" result="text"
import numpy as np
from PIL import Image
from skai.core import Video
from skai.extraction import ImageExtractor, MultimodalExtractor, VideoExtractor

scan = Image.new('RGB', (64, 48), 'white')
print(ImageExtractor(fields={'vendor': 'vendor name', 'total': 'total amount'}, backend='dummy').fit().predict([scan]))
video = Video(np.zeros((8, 8, 8, 3), dtype=np.uint8), fps=4)
print(VideoExtractor(fields={'place': 'where it happens'}, n_frames=4, backend='dummy').fit().predict([video]))
product = {'photo': Image.new('RGB', (8, 8)), 'notes': 'SkaiBox, blue'}
extractor = MultimodalExtractor(fields={'product': 'product name', 'color': 'color'}, backend='dummy')
print(extractor.fit().predict([product]))
```

## What fit learns

`fit(X, y)` with reference records as `y` passes `k_shot` sample and record pairs to the backend as examples, at most
`max_examples` when `k_shot` is `None`.

```python exec="true" source="above" result="text"
from skai.extraction import TextExtractor

extractor = TextExtractor(fields={'name': 'person name'}, backend='dummy')
print(extractor.fit(['Ada lives here.'], [{'name': 'Ada'}]).examples_)
```

The API reference is in [`skai.extraction`](../api/extraction/index.md).
