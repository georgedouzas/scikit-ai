# Media generation

The media generators of `skai.generation` return one image, audio sample or video per sample, of the type their name
ends with, in the requested `output_format`. Every generator takes an `operation` from its own set, validated at
`fit`. Fixed conditioning, such as a voice or a portrait, stays in constructor parameters, and conditioning that
changes per sample is a key of a multimodal sample.

| Class | `operation`, default first |
|---|---|
| `TextToImageGenerator` | `'generate'` |
| `ImageToImageGenerator` | `'edit'`, `'upscale'`, `'remove_background'`, `'collage'` |
| `VideoToImageGenerator` | `'thumbnail'`, `'frame'` |
| `MultimodalToImageGenerator` | `'edit'` |
| `TextToAudioGenerator` | `'speech'`, `'music'`, `'sound'` |
| `AudioToAudioGenerator` | `'separate'`, `'enhance'`, `'convert_voice'`, `'concatenate'` |
| `VideoToAudioGenerator` | `'extract'` |
| `MultimodalToAudioGenerator` | `'mix'`, `'speech'` |
| `TextToVideoGenerator` | `'generate'` |
| `ImageToVideoGenerator` | `'animate'` |
| `AudioToVideoGenerator` | `'talking_head'` |
| `VideoToVideoGenerator` | `'reframe'`, `'concatenate'` |
| `MultimodalToVideoGenerator` | `'lip_sync'`, `'mux'`, `'subtitle'` |

An operation that the class does not offer raises `InvalidParameterError` listing the valid ones, and an operation
that the backend does not support raises `UnsupportedBackendError` at `fit`, listing the backends that do.

## Local operations

`collage`, `frame`, `concatenate`, `extract`, `mix`, `mux` and `subtitle` are deterministic media operations that run
locally, without a backend or credentials. Video operations need the `video` extra. The first version holds whole
videos in memory, so long videos need much memory.

## Images

`TextToImageGenerator` creates one image of `size` per prompt, with an optional `style` and `negative_prompt`.
`reference_images` show a subject that every image keeps consistent, on backends that take reference images.
`ImageToImageGenerator` edits every image with one fixed `instructions` inside an optional `mask`, upscales by
`scale`, removes the background into a transparent image, or lays a list of images out as a collage.

```python exec="true" source="above" result="text"
from PIL import Image
from skai.generation import ImageToImageGenerator, TextToImageGenerator

generator = TextToImageGenerator(size=(64, 48), style='watercolor', random_state=0, backend='dummy')
image = generator.fit().predict(['a lake'])[0]
edited = ImageToImageGenerator(instructions='add snow', backend='dummy').fit().predict([image])[0]
upscaled = ImageToImageGenerator(operation='upscale', scale=2, backend='dummy').fit().predict([image])[0]
cutout = ImageToImageGenerator(operation='remove_background', backend='dummy').fit().predict([image])[0]
collage = ImageToImageGenerator(operation='collage', tile=32).fit().predict([[image, edited, upscaled]])[0]
print(edited.size, upscaled.size, cutout.mode, collage.size)
```

`VideoToImageGenerator` returns the thumbnail that the backend picks among `n_frames` sampled frames, or the frame at
`at` seconds. `MultimodalToImageGenerator` edits the `image` of every sample with the `instructions` of the same
sample, inside its optional `mask`, so every image gets its own edit.

```python exec="true" source="above" result="text"
import numpy as np
from PIL import Image
from skai.core import Video
from skai.generation import MultimodalToImageGenerator, VideoToImageGenerator

video = Video(np.zeros((8, 48, 64, 3), dtype=np.uint8), fps=4)
thumbnail = VideoToImageGenerator(n_frames=4, backend='dummy').fit().predict([video])[0]
frame = VideoToImageGenerator(operation='frame', at=1.0).fit().predict([video])[0]
mask = Image.new('L', (64, 48), 255)
sample = {'image': Image.new('RGB', (64, 48)), 'instructions': 'Write Γεια here.', 'mask': mask}
print(thumbnail.size, frame.size, MultimodalToImageGenerator(backend='dummy').fit().predict([sample])[0].size)
```

## Audio

`TextToAudioGenerator` speaks every text in `voice` and `language`, or creates music or a sound of `duration`
seconds that the text describes. A string `voice` names a voice of the backend, and reference audio clones a voice on
backends that support it.

```python exec="true" source="above" result="text"
from skai.generation import TextToAudioGenerator

speech = TextToAudioGenerator(voice='alloy', backend='dummy').fit().predict(['Welcome to scikit-ai.'])[0]
music = TextToAudioGenerator(operation='music', duration=2, backend='dummy').fit().predict(['calm piano'])[0]
sound = TextToAudioGenerator(operation='sound', duration=1, backend='dummy').fit().predict(['a door slams'])[0]
print(speech, music, sound)
```

`AudioToAudioGenerator` returns the `stem` of a separation, `'vocals'` or `'accompaniment'`, a denoised copy, the
same speech in another `voice`, or a list of audio samples joined with `gap` seconds between them.
`VideoToAudioGenerator` returns the mono audio track of every video, or silence as long as a video without one.

```python exec="true" source="above" result="text"
import numpy as np
from skai.core import Video
from skai.generation import AudioToAudioGenerator, VideoToAudioGenerator

song = (0.3 * np.sin(np.arange(32000) / 10), 16000)
vocals = AudioToAudioGenerator(operation='separate', stem='vocals', backend='dummy').fit().predict([song])[0]
clean = AudioToAudioGenerator(operation='enhance', backend='dummy').fit().predict([song])[0]
converted = AudioToAudioGenerator(operation='convert_voice', voice=song, backend='dummy').fit().predict([song])[0]
joined = AudioToAudioGenerator(operation='concatenate', gap=0.5).fit().predict([[song, song]])[0]
track = VideoToAudioGenerator().fit().predict([Video(np.zeros((8, 8, 8, 3), dtype=np.uint8), fps=4)])[0]
print(vocals, clean, converted, joined, track)
```

`MultimodalToAudioGenerator(operation='mix')` mixes the named tracks of every sample into one track as long as the
longest. `volumes` sets the gain of every track, and the tracks in `duck_under` are lowered to `duck_level` while the
track `duck_for` sounds. A track is audio, a video whose audio track is used, or a list of timed records with `start`
and `media`, such as segmenter regions, each placed at its start. With `operation='speech'`, it speaks the `text` of
every sample in the `voice` of the same sample, a voice name or reference audio.

```python exec="true" source="above" result="text"
import numpy as np
from skai.generation import MultimodalToAudioGenerator

voice = (0.3 * np.sin(np.arange(16000) / 10), 16000)
music = (0.1 * np.sin(np.arange(48000) / 30), 16000)
mixer = MultimodalToAudioGenerator(volumes={'music': 0.5}, duck_under=['music'], duck_for='voice').fit()
print(mixer.predict([{'voice': voice, 'music': music, 'effect': [{'start': 2.0, 'media': voice}]}]))
speaker = MultimodalToAudioGenerator(operation='speech', backend='dummy').fit()
print(speaker.predict([{'text': 'Hello.', 'voice': 'alloy'}, {'text': 'Hi there.', 'voice': voice}]))
```

## Video

`TextToVideoGenerator` creates one video per prompt, of `duration` seconds and `size` pixels. `ImageToVideoGenerator`
turns every image into a short video that starts from it, with optional `instructions` for the motion.
`AudioToVideoGenerator` makes a fixed `avatar` speak every audio sample, as a talking head with the audio as its
track. Backends with fixed video lengths or sizes, such as Sora, generate the closest and trim and resize the result,
which needs the `video` extra.

```python exec="true" source="above" result="text"
import numpy as np
from PIL import Image
from skai.generation import AudioToVideoGenerator, ImageToVideoGenerator, TextToVideoGenerator

video = TextToVideoGenerator(duration=2, size=(64, 48), fps=8, backend='dummy').fit().predict(['a boat at sea'])[0]
animator = ImageToVideoGenerator(instructions='slow zoom in', duration=1, fps=8, backend='dummy').fit()
talking = AudioToVideoGenerator(avatar=Image.new('RGB', (64, 64), 'beige'), backend='dummy').fit()
print(video, animator.predict([Image.new('RGB', (64, 48))])[0], talking.predict([(np.zeros(16000), 16000)])[0])
```

`VideoToVideoGenerator(operation='reframe')` crops every video to `aspect_ratio`, following the `subject` that the
backend detects in `n_frames` sampled frames, and `operation='concatenate'` joins a list of clips at the size and frame
rate of the first. `MultimodalToVideoGenerator` lip-syncs `video` to `audio`, replaces the soundtrack of `video` with
`audio`, or shows `transcript` as subtitles. A transcript is a string, or a list of timed records with `start`, `end`
and `text`, such as the regions of `AudioSegmenter`. Subtitles are burned into the frames, or with `burn=False`
attached as `subtitles` in SRT format and written next to the video.

```python exec="true" source="above" result="text"
import numpy as np
from skai.core import Video
from skai.generation import MultimodalToVideoGenerator, VideoToVideoGenerator

clip = Video(np.zeros((8, 48, 64, 3), dtype=np.uint8), fps=4)
speech = (np.zeros(32000), 16000)
vertical = VideoToVideoGenerator(aspect_ratio=9 / 16, backend='dummy').fit().predict([clip])[0]
joined = VideoToVideoGenerator(operation='concatenate').fit().predict([[clip, clip, clip]])[0]
synced = MultimodalToVideoGenerator(backend='dummy').fit().predict([{'video': clip, 'audio': speech}])[0]
muxed = MultimodalToVideoGenerator(operation='mux').fit().predict([{'video': clip, 'audio': speech}])[0]
subtitler = MultimodalToVideoGenerator(operation='subtitle', burn=False).fit()
subtitled = subtitler.predict([{'video': clip, 'transcript': [{'start': 0.0, 'end': 1.5, 'text': 'Hello.'}]}])[0]
print(vertical.frames.shape, joined, synced, muxed.audio)
print(subtitled.subtitles)
```

## Responsible use

Cloned voices, converted voices, lip sync and talking heads can imitate real people. Generate them only with the
consent of the people they depict, label them as generated where they are shared, and follow the usage policies of
the provider that runs the model.

## What fit learns

Media generators prepare their conditioning in `fit`:

- `TextToAudioGenerator` and `AudioToAudioGenerator(operation='convert_voice')` join one or more recordings in `voice`
  into `voice_`. Backends that cannot clone voices raise `UnsupportedBackendError` at `fit`.
- `TextToImageGenerator` learns `style_` from `style_images`, written by the backend's image description, and an
  explicit `style` takes precedence. With `reference_prompt`, it generates `reference_images_` and sends them with
  every request, and explicit `reference_images` take precedence.
- `AudioToVideoGenerator` prepares `avatar_`: converted to RGB, center-cropped to a square, and resized to the input
  size of the backend.
- `ImageToImageGenerator` reads `mask` into `mask_`.

```python exec="true" source="above" result="text"
from PIL import Image
from skai.generation import TextToImageGenerator

references = [Image.new('RGB', (16, 16), color) for color in ('teal', 'navy')]
styled = TextToImageGenerator(size=(64, 48), style_images=references, backend='dummy').fit()
consistent = TextToImageGenerator(size=(64, 48), reference_prompt='a girl with a red scarf', backend='dummy').fit()
print(styled.style_, len(consistent.reference_images_))
```

The API reference is in [`skai.generation`](../api/generation/index.md).
