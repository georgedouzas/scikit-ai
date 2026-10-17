# Audio components

The components of `skai.audio` take audio as input, in any form listed in [Media inputs and outputs](media.md).

## Transcribe speech

`SpeechTranscriber` returns one transcript per audio sample, in an optional `language`. With `timestamps='segment'` or
`timestamps='word'`, `predict` returns records with `start`, `end` and `text`, while `transform` still returns the
plain text so the transcriber can feed a text component. A backend without the requested granularity raises
`UnsupportedBackendError` at `fit`.

```python exec="true" source="above" result="text"
import numpy as np
from skai.audio import SpeechTranscriber

audio = (np.zeros(32000, dtype=np.float32), 16000)
print(SpeechTranscriber(backend='dummy').fit().predict([audio]))
print(SpeechTranscriber(timestamps='segment', backend='dummy').fit().predict([audio])[0])
```

## Animate a talking portrait

`TalkingAvatarGenerator` animates a fixed `avatar` so that it speaks every audio sample, and returns one lip-synced
video per sample with the audio as its track. With `SpeechSynthesizer` in front of it, a pipeline turns texts into
talking videos.

```python exec="true" source="above" result="text"
from PIL import Image
from sklearn.pipeline import make_pipeline
from skai.audio import TalkingAvatarGenerator
from skai.text import SpeechSynthesizer

pipeline = make_pipeline(
    SpeechSynthesizer(backend='dummy'),
    TalkingAvatarGenerator(avatar=Image.new('RGB', (64, 64), 'beige'), backend='dummy'),
)
videos = pipeline.fit(None).predict(['Hello!', 'Thank you for watching.'])
print([round(video.duration, 2) for video in videos])
```

## Responsible use

Cloned voices and talking avatars can imitate real people. Generate them only with the consent of the people they
depict, label them as generated where they are shared, and follow the usage policies of the provider that runs the
model.

The API reference of every component is in [`skai.audio`](../api/audio/index.md).
