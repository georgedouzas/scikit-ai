# Segmentation

`skai.segmentation` holds the components that split every sample into labeled regions in space or time:
`TextSegmenter`, `ImageSegmenter`, `AudioSegmenter` and `VideoSegmenter`. Each returns one list of region dicts per
sample, and `by` chooses what to find. Every region carries `label`, `score` and `media`, the cropped piece of the
sample, so later steps can work on every region with `ForEach` and generators can join the pieces again. A `by` that
the input does not offer raises `InvalidParameterError` at `fit`.

| Class | `by` | Region keys besides `label`, `score` and `media` | Runs |
|---|---|---|---|
| `TextSegmenter` | `'paragraph'` (default), `'sentence'`, `'chunk'` | `start`, `end` character offsets | locally |
| `ImageSegmenter` | `'object'` (default) | `box` | `detect_objects` |
| `ImageSegmenter` | `'mask'` | `box`, `mask` | `segment_image` |
| `AudioSegmenter` | `'speech'` (default), `'word'` | `start`, `end`, `text` | `transcribe` |
| `AudioSegmenter` | `'speaker'` | `start`, `end`, `speaker`, `text` | `diarize` |
| `AudioSegmenter` | `'silence'` | `start`, `end` | locally |
| `AudioSegmenter` | `'moment'` | `start`, `end` | `find_moments` |
| `VideoSegmenter` | `'shot'` (default) | `start`, `end` | locally |
| `VideoSegmenter` | `'moment'` | `start`, `end` | `find_moments` |

Times are in seconds. Local choices need no backend and no model.

## Texts

`TextSegmenter` splits at blank lines, after sentence punctuation, or into chunks of `chunk_size` characters that
join back to the text.

```python exec="true" source="above" result="text"
from skai.segmentation import TextSegmenter

text = 'scikit-ai splits texts into chunks. They join back to the text.'
regions = TextSegmenter(by='chunk', chunk_size=25).fit().predict([text])[0]
print([(region['start'], region['end'], region['media']) for region in regions])
```

## Images

`ImageSegmenter(by='object')` returns boxes as `(left, top, right, bottom)` pixels, and `by='mask'` also returns
`mask`, a grayscale image of the input size whose white pixels belong to the region. The regions are named by
`labels` or described by `prompt`. A mask can feed the edit of `MultimodalToImageGenerator`.

```python exec="true" source="above" result="text"
from PIL import Image
from skai.segmentation import ImageSegmenter

photo = Image.new('RGB', (64, 48), 'skyblue')
objects = ImageSegmenter(labels=['person'], backend='dummy').fit().predict([photo])[0]
masks = ImageSegmenter(by='mask', prompt='the sky', backend='dummy').fit().predict([photo])[0]
print([(region['label'], region['box']) for region in objects], masks[0]['mask'].size)
```

## Audio

`AudioSegmenter` returns timed speech, timed words, speaker turns, the parts between silences, or the moments that
match `prompt`. Timed speech replaces the timestamps of the 0.2 transcribers.

```python exec="true" source="above" result="text"
import numpy as np
from skai.segmentation import AudioSegmenter

recording = (0.3 * np.sin(np.arange(32000) / 10), 16000)
for by in ('speech', 'word', 'speaker'):
    regions = AudioSegmenter(by=by, backend='dummy').fit().predict([recording])[0]
    print(by, [(region['start'], region['end'], region['text']) for region in regions][:2])
print(AudioSegmenter(by='silence').fit().predict([recording])[0][0]['media'])
```

## Videos

`VideoSegmenter(by='shot')` finds cuts from the color histograms of consecutive frames, without a model, and
`by='moment'` asks the backend for the parts that match `prompt`.

```python exec="true" source="above" result="text"
import numpy as np
from skai.core import Video
from skai.segmentation import VideoSegmenter

frames = np.concatenate([np.zeros((4, 8, 8, 3)), np.full((4, 8, 8, 3), 255)]).astype(np.uint8)
shots = VideoSegmenter(by='shot').fit().predict([Video(frames, fps=4)])[0]
print([(shot['start'], shot['end'], shot['media']) for shot in shots])
```

## What fit learns

With `by='speech'` or `by='word'`, `AudioSegmenter.fit(X, y)` with reference transcripts as `y` learns `vocabulary_`,
the domain terms of the references, as `AudioToTextGenerator` does, and sends it with every request. The other
segmenters learn nothing from data.

```python exec="true" source="above" result="text"
import numpy as np
from skai.segmentation import AudioSegmenter

segmenter = AudioSegmenter(backend='dummy').fit([(np.zeros(16000), 16000)], ['We deploy Kubernetes over gRPC.'])
print(segmenter.vocabulary_)
```

The API reference is in [`skai.segmentation`](../api/segmentation/index.md).
