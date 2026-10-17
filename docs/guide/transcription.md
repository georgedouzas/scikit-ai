# Transcription

`skai.transcription` holds the components that turn speech into text. `SpeechTranscriber` returns one transcript per
audio sample, in an optional `language`. With `timestamps='segment'` or `timestamps='word'`, `predict` returns records
with `start`, `end` and `text`, while `transform` still returns the plain text so the transcriber can feed a text
component. A backend without the requested granularity raises `UnsupportedBackendError` at `fit`.

```python exec="true" source="above" result="text"
import numpy as np
from skai.transcription import SpeechTranscriber

audio = (np.zeros(32000, dtype=np.float32), 16000)
print(SpeechTranscriber(backend='dummy').fit().predict([audio]))
print(SpeechTranscriber(timestamps='segment', backend='dummy').fit().predict([audio])[0])
```

## Videos

`VideoTranscriber` transcribes the audio track of every video, with the parameters and outputs of
`SpeechTranscriber`. Timestamps are measured from the start of the video, and a video without an audio track gives
an empty transcript without calling the backend.

```python exec="true" source="above" result="text"
import numpy as np
from skai.core import Audio, Video
from skai.transcription import VideoTranscriber

speaking = Video(np.zeros((20, 8, 8, 3), dtype=np.uint8), fps=10, audio=Audio(np.zeros(32000), 16000))
silent = Video(np.zeros((20, 8, 8, 3), dtype=np.uint8), fps=10)
print(VideoTranscriber(backend='dummy').fit().predict([speaking, silent]))
```

## What fit learns

`fit(X, y)` with reference transcripts as `y` learns `vocabulary_`, the names, products and technical terms of the
references, which the backend receives as context for spelling.

```python exec="true" source="above" result="text"
import numpy as np
from skai.transcription import SpeechTranscriber

audio = [(np.zeros(16000, dtype=np.float32), 16000)]
transcriber = SpeechTranscriber(backend='dummy').fit(audio, ['We deploy Kubernetes over gRPC.'])
print(transcriber.vocabulary_)
```

The API reference is in [`skai.transcription`](../api/transcription/index.md).
