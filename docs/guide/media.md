# Media inputs and outputs

Every component takes `X` as a container of samples and returns one output per sample, in input order. The class
name fixes the type of every sample, so a string is a text in a `Text` class and a path or URL in a media class.

## Sample containers

`X` can be a list, a tuple, a one-dimensional NumPy array, a pandas Series, or a single-column two-dimensional array or
DataFrame. Each element is one sample. An input with more than one column raises `ValueError`, except in `Multimodal`
classes, which read every row of a DataFrame as one sample.

```python exec="true" source="above" result="text"
import pandas as pd
from skai.generation import TextToTextGenerator

summarizer = TextToTextGenerator(instructions='Summarize the text.', max_length=2, backend='dummy').fit()
print(summarizer.predict(pd.DataFrame({'text': ['First long text.', 'Second long text.']})))
```

## Accepted samples

| Input | Accepted forms |
|---|---|
| Text | `str` |
| Image | local path, `http(s)` URL, encoded bytes, PIL image, NumPy array of shape `(H, W)` or `(H, W, C)` |
| Audio | local path, `http(s)` URL, encoded bytes, `skai.core.Audio`, `(samples, sample_rate)` tuple |
| Video | local path, `http(s)` URL, encoded bytes, `skai.core.Video` |
| Multimodal | a `dict` of named texts, images, audio, videos or transcripts, or a DataFrame row of them |

A text sample that is not a string raises `skai.core.InvalidSampleError`, a `TypeError`. A media sample that cannot be
read or decoded raises `skai.core.MediaReadError`, a `ValueError`. Both name the index of the sample. With
`on_error='ignore'`, an unreadable media sample gives a missing output and a warning instead.

```python exec="true" source="above" result="text"
import numpy as np
from PIL import Image
from skai.generation import ImageToTextGenerator

describer = ImageToTextGenerator(instructions='Describe the image.', backend='dummy').fit()
print(describer.predict([Image.new('RGB', (64, 48)), np.zeros((16, 16), dtype=np.uint8)]))
```

Video classes sample `n_frames` frames evenly over every video, and send the audio track to backends that accept
audio.

## Multimodal samples

A `Multimodal` class reads every value of a sample by its own kind: strings are texts unless they are URLs or paths of
existing media files, and the other values are read as images, audio or videos. The backend receives every part in one
request, each named by its key. An operation that needs specific keys, such as `video` and `audio` for a lip sync,
raises `InvalidSampleError` naming the missing key and the sample index.

```python exec="true" source="above" result="text"
import pandas as pd
from PIL import Image
from skai.classification import MultimodalClassifier
from skai.generation import MultimodalToTextGenerator

posts = pd.DataFrame({'image': [Image.new('RGB', (8, 8)), Image.new('RGB', (8, 8), 'red')], 'caption': ['hi', 'sale']})
print(MultimodalClassifier(labels=['safe', 'unsafe'], backend='dummy').fit().predict(posts))
captioner = MultimodalToTextGenerator(instructions='Write a caption.', backend='dummy').fit()
print(captioner.predict([{'photo': Image.new('RGB', (8, 8)), 'notes': 'taken in Crete'}]))
```

`MediaUnion` builds multimodal samples inside a pipeline, as [Pipelines](pipelines.md) shows.

## List-valued samples

Segmenters return one list of regions per sample. Only `ForEach`, `TopK` and the list operations of
generators, `collage` and `concatenate`, accept list-valued samples. Any other component raises `InvalidSampleError`
and suggests wrapping the step in `ForEach`.

```python exec="true" source="above" result="text"
from skai.core import InvalidSampleError
from skai.generation import TextToTextGenerator
from skai.pipeline import ForEach

generator = TextToTextGenerator(instructions='Shorten the text.', max_length=1, backend='dummy')
try:
    generator.fit().predict([['First passage', 'Second passage']])
except InvalidSampleError as error:
    print(error)
print(ForEach(generator).fit_transform([['First passage', 'Second passage']]))
```

## Outputs

Text outputs are one-dimensional NumPy arrays of `dtype=object` that hold strings. Classifiers return labels that follow
the targets seen in `fit`, embedders return float arrays of shape `(n_samples, n_dimensions)`, extractors return one
dict per sample, and segmenters return one list of region dicts per sample.

Components that generate images, audio or video take `output_format`:

| `output_format` | Images | Audio | Video |
|---|---|---|---|
| `'object'` (default) | PIL image | `skai.core.Audio` | `skai.core.Video` |
| `'bytes'` | PNG bytes | WAV bytes | MP4 bytes |
| `'path'` | PNG file | WAV file | MP4 file |

With `output_format='path'`, every output is written to a new file in `output_dir`, or in a new temporary directory when
`output_dir` is `None`. Existing files are never overwritten. Encoding video needs the `video` extra.

```python exec="true" source="above" result="text"
import tempfile
from skai.generation import TextToImageGenerator

generator = TextToImageGenerator(size=(64, 48), output_format='path', output_dir=tempfile.mkdtemp(), backend='dummy')
paths = generator.fit().predict(['a red square', 'a blue square'])
print(len(paths), paths[0].endswith('.png'))
```

`Audio` holds the `samples` array and the `sample_rate`, and `Video` holds the `frames`, the `fps`, an optional
`audio` track and optional `subtitles`. Both have `save` and `to_bytes` methods, and both are accepted back as inputs
by any component.

```python exec="true" source="above" result="text"
from skai.generation import AudioToTextGenerator, TextToAudioGenerator

speech = TextToAudioGenerator(backend='dummy').fit().predict(['Hello there'])
print(speech[0], AudioToTextGenerator(backend='dummy').fit().predict(speech))
```
