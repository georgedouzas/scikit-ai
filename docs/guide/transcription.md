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

The API reference is in [`skai.transcription`](../api/transcription/index.md).
