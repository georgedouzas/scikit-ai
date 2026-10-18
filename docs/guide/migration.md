# Migrating from 0.2

Version 0.3 replaces the task-named classes of 0.2 with a grid of classes named by input and output, described in the
[user guide](index.md#the-naming-grid). Tasks that are prompts became `instructions`, and model families or procedures
became `operation`. 0.3 has no aliases, so every removed name raises `ImportError`.

## Unchanged names

`TextClassifier`, `ImageClassifier`, `AudioClassifier`, `VideoClassifier`, `TextEmbedder`, `ImageEmbedder` and
`AudioEmbedder` keep their names, modules and parameters. `TextExtractor` keeps its name and parameters, and moves
from `skai.text` to `skai.extraction`.

## Removed modules

| 0.2 module | 0.3 module |
|---|---|
| `skai.text` | `skai.generation` for summaries and translations, `skai.extraction` for `TextExtractor` |
| `skai.description` | `skai.generation` |
| `skai.transcription` | `skai.generation` for transcripts, `skai.segmentation` for timestamps |

## Removed classes

| 0.2 | 0.3 |
|---|---|
| `TextSummarizer(max_length=50, style='one line')` | `TextToTextGenerator(instructions='Summarize the text in one line.', max_length=50)` or `presets.summarizer(max_length=50)` |
| `TextSummarizer(max_length=None)` | `TextToTextGenerator(instructions='Summarize the text.', max_length='auto')` |
| `TextTranslator(target_language='Greek')` | `TextToTextGenerator(instructions='Translate into Greek.', learn_terms=True)` or `presets.translator('Greek')` |
| `TextTranslator(target_language='Greek', source_language='English')` | `TextToTextGenerator(instructions='Translate from English into Greek.', learn_terms=True)` |
| `ImageDescriptor(detail='detailed', instructions='Mention the colors.')` | `ImageToTextGenerator(instructions='Describe the image in detail. Mention the colors.')` or `presets.describer()` |
| `AudioDescriptor()` | `AudioToTextGenerator(operation='instruct', instructions='Describe the audio.')` |
| `VideoDescriptor(n_frames=6)` | `VideoToTextGenerator(instructions='Describe the video.', n_frames=6)` |
| `SpeechTranscriber(language='en')` | `AudioToTextGenerator(language='en')`, whose default operation is `'transcribe'` |
| `SpeechTranscriber(timestamps='segment')` | `AudioSegmenter(by='speech')`, whose regions carry `start`, `end` and `text` |
| `SpeechTranscriber(timestamps='word')` | `AudioSegmenter(by='word')` |
| `VideoTranscriber()` | `VideoToTextGenerator(operation='transcribe')` |
| `VideoTranscriber(timestamps='segment')` | `make_pipeline(VideoToAudioGenerator(), AudioSegmenter(by='speech'))` |
| `SpeechSynthesizer(voice='alloy')` | `TextToAudioGenerator(voice='alloy')` |
| `ImageGenerator(size=(512, 512), style='watercolor')` | `TextToImageGenerator(size=(512, 512), style='watercolor')` |
| `ImageEditor(instruction='make it night', mask=mask)` | `ImageToImageGenerator(instructions='make it night', mask=mask)` |
| `ImageAnimator(motion_prompt='slow zoom', duration=2)` | `ImageToVideoGenerator(instructions='slow zoom', duration=2)` |
| `VideoGenerator(duration=4, size=(1280, 720))` | `TextToVideoGenerator(duration=4, size=(1280, 720))` |
| `TalkingAvatarGenerator(avatar=portrait)` | `AudioToVideoGenerator(avatar=portrait)` or `presets.talking_head(portrait)` |

The parameters that 0.2 classes shared, such as `backend`, `k_shot`, `adaptation`, `use_y`, `output_format` and
`random_state`, keep their meaning. The learned attributes keep their names: `examples_`, `max_length_`, `glossary_`,
`vocabulary_`, `voice_`, `style_` and `avatar_`. With `max_length`, `None` now sets no limit and `'auto'` learns the
length from the references. `ImageEditor` took `instruction`, and `ImageToImageGenerator` takes `instructions`.

```python exec="true" source="above" result="text"
import numpy as np
from sklearn.pipeline import make_pipeline
from skai.core import Audio, Video
from skai.generation import AudioToTextGenerator, TextToTextGenerator, VideoToAudioGenerator
from skai.segmentation import AudioSegmenter

summarizer = TextToTextGenerator(instructions='Summarize the text in one line.', max_length=3, backend='dummy')
translator = TextToTextGenerator(instructions='Translate into Greek.', learn_terms=True, backend='dummy')
print(summarizer.fit().predict(['A long text about scikit-ai.']), translator.fit().predict(['Good morning']))

audio = Audio(np.zeros(32000, dtype=np.float32), 16000)
print(AudioToTextGenerator(backend='dummy').fit().predict([audio]))
print(AudioSegmenter(by='speech', backend='dummy').fit().predict([audio])[0][0]['text'])
video = Video(np.zeros((16, 8, 8, 3), dtype=np.uint8), fps=8, audio=audio)
timed = make_pipeline(VideoToAudioGenerator(), AudioSegmenter(by='speech', backend='dummy'))
print([(region['start'], region['end']) for region in timed.fit(None).predict([video])[0]])
```

0.2 pipelines keep their shape with the new names:

```python exec="true" source="above" result="text"
from PIL import Image
from sklearn.pipeline import make_pipeline
from skai.generation import AudioToVideoGenerator, TextToAudioGenerator, TextToTextGenerator

presenter = make_pipeline(
    TextToTextGenerator(instructions='Translate into Spanish.', backend='dummy'),
    TextToAudioGenerator(backend='dummy'),
    AudioToVideoGenerator(avatar=Image.new('RGB', (32, 32), 'beige'), backend='dummy'),
)
print(presenter.fit(None).predict(['Welcome!', 'Thanks for watching.']))
```

## Custom backends

The backend tasks `summarize`, `translate`, `describe_image`, `describe_audio` and `describe_video` merge into one
`generate_text` task, and their request types merge into `TextGenerationRequest`:

| 0.2 | 0.3 |
|---|---|
| `summarize(SummaryRequest)` | `generate_text(TextGenerationRequest)` |
| `translate(TranslationRequest)` | `generate_text(TextGenerationRequest)` |
| `describe_image(DescriptionRequest)`, `describe_audio(...)`, `describe_video(...)` | `generate_text(TextGenerationRequest)` |
| `request.text` | `request.sample`, a text, image, audio, sampled video frames or multimodal sample |
| `request.style`, `request.target_language`, `request.source_language`, `request.detail` | `request.instructions` |
| `request.examples`, `request.max_length`, `request.glossary` | the same fields |

A `PromptedBackend` subclass needs no change, since it builds `generate_text` from `complete`. A `Backend` subclass
that implemented the 0.2 tasks implements `generate_text` instead, as [Backends](backends.md#registering-a-backend)
shows. The new operations add the tasks listed in [Backends](backends.md#tasks), and `supports(task, input_kind)`
says whether a backend runs a task on an input kind.

## Migrating from 0.1

Version 0.2 refocused scikit-ai on task-oriented components for text, images, audio and video. The `skai.llm` and
`skai.base` packages are removed, and components live in modules named after their task family, such as
`skai.classification` and `skai.generation`.

`skai.llm.OpenAIClassifier` is replaced by `skai.classification.TextClassifier` with the OpenAI backend:

| 0.1 | 0.2 and later |
|---|---|
| `OpenAIClassifier()` | `TextClassifier(backend='openai')` |
| `OpenAIClassifier(classes={0: 'negative', 1: 'positive'})` | `TextClassifier(labels={0: 'negative', 1: 'positive'}, backend='openai')` |
| `OpenAIClassifier(k_shot=3)` | `TextClassifier(k_shot=3, backend='openai')` |
| `OpenAIClassifier(openai_client=client)` | `TextClassifier(backend=OpenAIBackend(client=client))` |
| `OpenAIClassifier(responses_kwargs={...})` | `TextClassifier(backend=OpenAIBackend(options={...}))` |

Constructor parameters are keyword-only. `k_shot=None` selects one example per class, as in 0.1, and `k_shot=0` is
zero-shot. The API key is read from `OPENAI_API_KEY`, or from the variable named by
`OpenAIBackend(api_key_env=...)`, and `.env` files are no longer loaded automatically.

```python exec="true" source="above" result="text"
from skai.backends import DummyBackend
from skai.classification import TextClassifier

classifier = TextClassifier(labels={0: 'negative', 1: 'positive'}, backend=DummyBackend())
print(classifier.fit().predict(['I loved it.']))
```
