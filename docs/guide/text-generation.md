# Text generation

The text generators of `skai.generation` return one text per sample, from both `predict` and `transform`:
`TextToTextGenerator`, `ImageToTextGenerator`, `AudioToTextGenerator`, `VideoToTextGenerator` and
`MultimodalToTextGenerator`. Summaries, translations, descriptions, answers, scripts and transcripts are all text
generators, configured by their `operation` and `instructions`.

| Class | `operation` | Default | Does |
|---|---|---|---|
| `TextToTextGenerator` | `'instruct'` | `'instruct'` | follows `instructions` |
| `ImageToTextGenerator` | `'instruct'`, `'transcribe'` | `'instruct'` | follows `instructions`, or returns the written text |
| `AudioToTextGenerator` | `'transcribe'`, `'instruct'` | `'transcribe'` | returns the spoken text, or follows `instructions` |
| `VideoToTextGenerator` | `'instruct'`, `'transcribe'` | `'instruct'` | follows `instructions`, or returns the spoken text |
| `MultimodalToTextGenerator` | `'instruct'` | `'instruct'` | follows `instructions` with every named part |

## Following instructions

With `operation='instruct'`, a text generator follows its required `instructions` with an instructable model, and
`max_length` cuts every output to a number of words. Fitting without `instructions` raises `InvalidParameterError`.

```python exec="true" source="above" result="text"
from PIL import Image
from skai.generation import ImageToTextGenerator, TextToTextGenerator

text = 'scikit-ai gives every media task the same scikit-learn interface, whatever model runs it.'
summarizer = TextToTextGenerator(instructions='Summarize in one sentence.', max_length=6, backend='dummy')
print(summarizer.fit().predict([text]))
print(TextToTextGenerator(instructions='Translate into Greek.', backend='dummy').fit().predict(['Good morning']))
describer = ImageToTextGenerator(instructions='Describe the image. Mention the colors.', backend='dummy')
print(describer.fit().predict([Image.new('RGB', (320, 240), 'navy')]))
```

`VideoToTextGenerator` sees `n_frames` frames sampled evenly over every video, and the audio track on backends that
accept audio. `AudioToTextGenerator(operation='instruct')` describes, summarizes or answers about audio with an
audio-language model. `MultimodalToTextGenerator` sends every named part of a sample in one request.

```python exec="true" source="above" result="text"
import numpy as np
from PIL import Image
from skai.core import Video
from skai.generation import AudioToTextGenerator, MultimodalToTextGenerator, VideoToTextGenerator

video = Video(np.zeros((30, 8, 8, 3), dtype=np.uint8), fps=10)
print(VideoToTextGenerator(instructions='Describe the video.', n_frames=6, backend='dummy').fit().predict([video]))
audio = AudioToTextGenerator(operation='instruct', instructions='Describe the sounds.', backend='dummy')
print(audio.fit().predict([(np.zeros(32000, dtype=np.float32), 16000)]))
captioner = MultimodalToTextGenerator(instructions='Write a caption.', backend='dummy').fit()
print(captioner.predict([{'photo': Image.new('RGB', (8, 8)), 'notes': 'taken in Crete'}]))
```

## Transcribing

With `operation='transcribe'`, `AudioToTextGenerator` (its default) and `VideoToTextGenerator` return exactly what is
said, in an optional `language`, with speech recognition models. A video without an audio track gives an empty
transcript without calling the backend. `ImageToTextGenerator` returns exactly what is written, as plain text or, with
`format='markdown'`, keeping the page structure. For timed transcripts, use `AudioSegmenter(by='speech')` or
`by='word'` from [Segmentation](segmentation.md).

```python exec="true" source="above" result="text"
import numpy as np
from PIL import Image
from skai.core import Audio, Video
from skai.generation import AudioToTextGenerator, ImageToTextGenerator, VideoToTextGenerator

audio = Audio(np.zeros(32000, dtype=np.float32), 16000)
print(AudioToTextGenerator(language='en', backend='dummy').fit().predict([audio]))
speaking = Video(np.zeros((20, 8, 8, 3), dtype=np.uint8), fps=10, audio=audio)
silent = Video(np.zeros((20, 8, 8, 3), dtype=np.uint8), fps=10)
print(VideoToTextGenerator(operation='transcribe', backend='dummy').fit().predict([speaking, silent]))
page = Image.new('RGB', (64, 48), 'white')
print(ImageToTextGenerator(operation='transcribe', format='markdown', backend='dummy').fit().predict([page]))
```

## What fit learns

With `operation='instruct'`, `fit(X, y)` with reference texts as `y` passes `k_shot` sample and reference pairs to the
backend as examples, so outputs follow their style, length and vocabulary. With `max_length='auto'`, it learns
`max_length_` from the 90th percentile of the reference word counts. `TextToTextGenerator(learn_terms=True)` learns
`glossary_`, the terms that appear in at least two source texts mapped to the term their references share, so they
are rendered consistently.

```python exec="true" source="above" result="text"
from skai.generation import TextToTextGenerator

summarizer = TextToTextGenerator(instructions='Summarize the text.', max_length='auto', backend='dummy')
summarizer.fit(['one two three four'], ['one two'])
translator = TextToTextGenerator(instructions='Translate into French.', learn_terms=True, backend='dummy')
translator.fit(['Open SkaiBox.', 'SkaiBox is ready.'], ['Ouvrez SkaiBox.', 'SkaiBox est prêt.'])
print(summarizer.max_length_, translator.glossary_)
```

With `operation='transcribe'`, `fit(X, y)` with reference transcripts as `y` learns `vocabulary_`, the names,
products and technical terms of the references, at most `max_vocabulary` of them, which the backend receives as
context for spelling. A backend that cannot take context warns at `fit`.

```python exec="true" source="above" result="text"
import numpy as np
from skai.generation import AudioToTextGenerator

audio = [(np.zeros(16000, dtype=np.float32), 16000)]
transcriber = AudioToTextGenerator(backend='dummy').fit(audio, ['We deploy Kubernetes over gRPC.'])
print(transcriber.vocabulary_)
```

Text generators with `operation='instruct'` also support `adaptation='finetune'`, described in
[What fit learns](fit.md).

The API reference is in [`skai.generation`](../api/generation/index.md).
