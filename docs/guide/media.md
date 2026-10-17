# Media inputs and outputs

Every component takes `X` as a container of samples and returns one output per sample, in input order.

## Sample containers

`X` can be a list, a tuple, a one-dimensional NumPy array, a pandas Series, or a single-column two-dimensional array or
DataFrame. Each element is one sample. An input with more than one column raises `ValueError`.

```python exec="true" source="above" result="text"
import pandas as pd
from skai.text import TextSummarizer

summarizer = TextSummarizer(max_length=2, backend='dummy').fit()
print(summarizer.predict(pd.DataFrame({'text': ['First long text.', 'Second long text.']})))
```

## Accepted samples

| Input | Accepted forms |
|---|---|
| Text | `str` |
| Image | local path, `http(s)` URL, encoded bytes, PIL image, NumPy array of shape `(H, W)` or `(H, W, C)` |
| Audio | local path, `http(s)` URL, encoded bytes, `skai.core.Audio`, `(samples, sample_rate)` tuple |
| Video | local path, `http(s)` URL, encoded bytes, `skai.core.Video` |

A text sample that is not a string raises `skai.core.InvalidSampleError`, a `TypeError`. A media sample that cannot be
read or decoded raises `skai.core.MediaReadError`, a `ValueError`. Both name the index of the sample. With
`on_error='ignore'`, an unreadable media sample gives a missing output and a warning instead.

```python exec="true" source="above" result="text"
import numpy as np
from PIL import Image
from skai.description import ImageDescriptor

descriptor = ImageDescriptor(backend='dummy').fit()
print(descriptor.predict([Image.new('RGB', (64, 48)), np.zeros((16, 16), dtype=np.uint8)]))
```

## Outputs

Text outputs are one-dimensional NumPy arrays of `dtype=object` that hold strings. Classifiers return labels that follow
the targets seen in `fit`, and embedders return float arrays of shape `(n_samples, n_dimensions)`.

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
from skai.generation import ImageGenerator

generator = ImageGenerator(size=(64, 64), output_format='path', output_dir=tempfile.mkdtemp(), backend='dummy')
paths = generator.fit().predict(['a red square', 'a blue square'])
print(len(paths), paths[0].endswith('.png'))
```

`Audio` holds the `samples` array and the `sample_rate`, and `Video` holds the `frames`, the `fps` and an optional
`audio` track. Both have `save` and `to_bytes` methods, and both are accepted back as inputs by any component.

```python exec="true" source="above" result="text"
from skai.transcription import SpeechTranscriber
from skai.generation import SpeechSynthesizer

speech = SpeechSynthesizer(backend='dummy').fit().predict(['Hello there'])
print(speech[0], SpeechTranscriber(backend='dummy').fit().predict(speech))
```
