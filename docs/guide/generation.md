# Generation

`skai.generation` holds the components that generate images, speech and video. Generators are named for what they
generate, take their input as `X`, keep fixed conditioning such as a voice or a portrait in their parameters, and
return media in the requested `output_format`.

## Speech and images

`SpeechSynthesizer` speaks every text in a `voice`. A string names a voice of the backend, and reference audio clones
a voice on backends that support it. `ImageGenerator` creates one image of `size` per prompt, with an optional `style`
and `negative_prompt`.

```python exec="true" source="above" result="text"
from skai.generation import ImageGenerator, SpeechSynthesizer

speech = SpeechSynthesizer(voice='alloy', backend='dummy').fit().predict(['Welcome to scikit-ai.'])
images = ImageGenerator(size=(256, 128), style='watercolor', random_state=0, backend='dummy').fit().predict(['a lake'])
print(speech[0], images[0].size)
```

## Talking portraits

`TalkingAvatarGenerator` animates a fixed `avatar` so that it speaks every audio sample, and returns one lip-synced
video per sample with the audio as its track. With `SpeechSynthesizer` in front of it, a pipeline turns texts into
talking videos.

```python exec="true" source="above" result="text"
from PIL import Image
from sklearn.pipeline import make_pipeline
from skai.generation import SpeechSynthesizer, TalkingAvatarGenerator

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

## What fit learns

Generators prepare their conditioning in `fit`. `SpeechSynthesizer` joins one or more recordings in `voice` into the
voice it clones. `ImageGenerator` learns `style_` from `style_images`, using the backend's image description, and an
explicit `style` takes precedence. `TalkingAvatarGenerator` crops the portrait to a centered square and resizes it to
the backend's input size.

```python exec="true" source="above" result="text"
from PIL import Image
from skai.generation import ImageGenerator

references = [Image.new('RGB', (16, 16), color) for color in ('teal', 'navy')]
generator = ImageGenerator(size=(64, 64), style_images=references, backend='dummy').fit()
print(generator.style_)
```

The API reference is in [`skai.generation`](../api/generation/index.md).
