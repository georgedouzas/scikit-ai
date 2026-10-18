[black badge]: <https://img.shields.io/badge/%20style-black-000000.svg>
[black]: <https://github.com/psf/black>
[docformatter badge]: <https://img.shields.io/badge/%20formatter-docformatter-fedcba.svg>
[docformatter]: <https://github.com/PyCQA/docformatter>
[ruff badge]: <https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v1.json>
[ruff]: <https://github.com/charliermarsh/ruff>
[mypy badge]: <http://www.mypy-lang.org/static/mypy_badge.svg>
[mypy]: <http://mypy-lang.org>
[properdocs badge]: <https://img.shields.io/badge/docs-properdocs-blue.svg?style=flat>
[properdocs]: <https://properdocs.org>
[pip-audit badge]: <https://img.shields.io/badge/security-pip--audit-green>
[pip-audit]: <https://github.com/pypa/pip-audit>
[bandit badge]: <https://img.shields.io/badge/security-bandit-yellow>
[bandit]: <https://github.com/PyCQA/bandit>
[pytest badge]: <https://img.shields.io/badge/tests-pytest-blue>
[pytest]: <https://github.com/pytest-dev/pytest>
[coverage badge]: <https://img.shields.io/badge/coverage-pytest--cov-blue>
[coverage]: <https://github.com/nedbat/coveragepy>
[interrogate badge]: <https://img.shields.io/badge/docstring-interrogate-blue>
[interrogate]: <https://github.com/econchick/interrogate>
[pre-commit badge]: <https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit>
[pre-commit]: <https://github.com/pre-commit/pre-commit>
[nox badge]: <https://img.shields.io/badge/%F0%9F%A6%8A-Nox-D85E00.svg>
[nox]: <https://github.com/wntrblm/nox>
[version badge]: <https://img.shields.io/pypi/v/scikit-ai.svg>
[pythonversion badge]: <https://img.shields.io/pypi/pyversions/scikit-ai.svg>
[downloads badge]: <https://img.shields.io/pypi/dd/scikit-ai>
[discussions]: <https://github.com/georgedouzas/scikit-ai/discussions>
[discussions badge]: <https://img.shields.io/github/discussions/georgedouzas/scikit-ai>
[ci]: <https://github.com/georgedouzas/scikit-ai/actions?query=workflow>
[ci badge]: <https://github.com/georgedouzas/scikit-ai/actions/workflows/ci.yml/badge.svg?branch=main>
[doc]: <https://github.com/georgedouzas/scikit-ai/actions?query=workflow>
[doc badge]: <https://github.com/georgedouzas/scikit-ai/actions/workflows/doc.yml/badge.svg?branch=main>

# scikit-ai

[![ci][ci badge]][ci] [![doc][doc badge]][doc]

| Category          | Tools    |
| ------------------| -------- |
| **Development**   | [![black][black badge]][black] [![ruff][ruff badge]][ruff] [![mypy][mypy badge]][mypy] [![docformatter][docformatter badge]][docformatter] |
| **Testing**       | [![pytest][pytest badge]][pytest] [![coverage][coverage badge]][coverage] [![interrogate][interrogate badge]][interrogate] |
| **Security**      | [![pip-audit][pip-audit badge]][pip-audit] [![bandit][bandit badge]][bandit] |
| **Automation**    | [![nox][nox badge]][nox] [![pre-commit][pre-commit badge]][pre-commit] |
| **Package**       | ![version][version badge] ![pythonversion][pythonversion badge] ![downloads][downloads badge] |
| **Documentation** | [![properdocs][properdocs badge]][properdocs]|
| **Communication** | [![discussions][discussions badge]][discussions] |

## Introduction

scikit-ai provides components for text, images, audio, video and multimodal samples that follow the
[scikit-learn](https://scikit-learn.org/stable) estimator API. Components are named by their input and output, such
as `ImageClassifier`, `AudioSegmenter` or `TextToVideoGenerator`, so the name says what every sample is and what every
output is. The same component runs on open-source models or on cloud APIs, chosen with one `backend` parameter, and
fits into `Pipeline`, `GridSearchCV` and the rest of the scikit-learn ecosystem. Presets and recipes configure the
components and pipelines of common tasks, from a summary to a video dubbed in the original voice.

## Installation

scikit-ai is available on PyPI. The base install includes the offline `dummy` backend, and every other backend is an
optional extra:

```bash
pip install scikit-ai              # components and the dummy backend
pip install "scikit-ai[hf]"        # local open-source models with Hugging Face
pip install "scikit-ai[openai]"    # the OpenAI API
pip install "scikit-ai[litellm]"   # Anthropic, Gemini, Mistral, Bedrock, Ollama and others through LiteLLM
pip install "scikit-ai[replicate]" # open-source models hosted by Replicate
pip install "scikit-ai[diarize]"   # speaker diarization with pyannote
pip install "scikit-ai[video]"     # encoding, decoding and assembling video
pip install "scikit-ai[all]"       # everything
```

Development installation requires to clone the repository and then use [PDM](https://github.com/pdm-project/pdm) to install the
project as well as the main and development dependencies:

```bash
git clone https://github.com/georgedouzas/scikit-ai.git
cd scikit-ai
pdm install
```

## Components

Components live in modules named after their role, like scikit-learn's estimators. Analyzers are named
`{Input}{Role}` and generators `{Input}To{Output}Generator`, where the input is `Text`, `Image`, `Audio`, `Video` or
`Multimodal`, a dict of named parts.

| Module | Components | Does | Output |
|---|---|---|---|
| `skai.classification` | `TextClassifier`, `ImageClassifier`, `AudioClassifier`, `VideoClassifier`, `MultimodalClassifier` | assign labels, zero-shot or few-shot | labels and probabilities |
| `skai.extraction` | `TextExtractor`, `ImageExtractor`, `AudioExtractor`, `VideoExtractor`, `MultimodalExtractor` | extract declared fields, nested records and lists | records or columns |
| `skai.embedding` | `TextEmbedder`, `ImageEmbedder`, `AudioEmbedder`, `VideoEmbedder` | map samples to vectors | float array |
| `skai.segmentation` | `TextSegmenter`, `ImageSegmenter`, `AudioSegmenter`, `VideoSegmenter` | split samples into regions in space or time | lists of regions |
| `skai.generation` | `TextToTextGenerator`, `ImageToTextGenerator`, `AudioToTextGenerator`, `VideoToTextGenerator`, `MultimodalToTextGenerator` | follow instructions, or transcribe speech and written text | text |
| `skai.generation` | `TextToImageGenerator`, `ImageToImageGenerator`, `VideoToImageGenerator`, `MultimodalToImageGenerator` | generate, edit, upscale, cut out, lay out and pick images | images |
| `skai.generation` | `TextToAudioGenerator`, `AudioToAudioGenerator`, `VideoToAudioGenerator`, `MultimodalToAudioGenerator` | speak, compose, separate, enhance, convert, join and mix audio | audio |
| `skai.generation` | `TextToVideoGenerator`, `ImageToVideoGenerator`, `AudioToVideoGenerator`, `VideoToVideoGenerator`, `MultimodalToVideoGenerator` | generate, animate, reframe, join, lip-sync, mux and subtitle video | videos |
| `skai.pipeline` | `MediaUnion`, `Field`, `ForEach`, `TopK`, `Retriever` | combine branches, pick fields, work on every item, keep the best, search | samples |
| `skai.presets`, `skai.recipes` | functions such as `summarizer` or `explainer_video` | configure components and pipelines for common tasks | estimators |

`operation` selects the procedure of a component, such as `'transcribe'` or `'upscale'`, and `instructions` holds the
prompt of the components that follow one, such as `'Summarize the text.'`.

## What fit learns

Every component learns from the data given to `fit`, without training a model unless you ask for fine-tuning.

<!-- --8<-- [start:fit] -->
| Component | `fit(X, y)` takes | Learns | Uses it as |
|---|---|---|---|
| classifiers | samples and labels | `classes_`, `examples_` | one example per class by default |
| extractors | samples and reference records | `examples_` | example pairs |
| text generators with `operation='instruct'` | samples and reference texts | `examples_`, `max_length_` | example pairs and, with `max_length='auto'`, a learned length |
| `TextToTextGenerator(learn_terms=True)` | texts and reference texts | `glossary_` | consistent terms |
| text generators with `operation='transcribe'`, `AudioSegmenter` | audio or video and reference transcripts | `vocabulary_` | domain terms sent as spelling context |
| embedders | samples | `mean_`, `components_` | centering, and PCA when the backend cannot shorten |
| `TextToAudioGenerator`, `AudioToAudioGenerator` | the `voice` parameter | `voice_` | a voice cloned from one or more recordings |
| `TextToImageGenerator` | the `style_images` and `reference_prompt` parameters | `style_`, `reference_images_` | a learned style and a consistent subject |
| `AudioToVideoGenerator` | the `avatar` parameter | `avatar_` | the portrait cropped and sized for the backend |

With `adaptation='finetune'`, classifiers and text generators with `operation='instruct'` train a model on all the
training data on backends that support fine-tuning, and `fit` raises `UnsupportedBackendError` on the others. In a
pipeline, intermediate steps are fitted through `fit_transform`, which ignores `y`, so a transcriber in front of a
classifier never reads the class labels as transcripts.
<!-- --8<-- [end:fit] -->

## Usage

Every component is a scikit-learn estimator. The examples use the offline `dummy` backend, so they run without models
or credentials. Replace `backend='dummy'` with a backend from the table below to get real results.

Classify texts with a few labeled examples:

```python
from skai.classification import TextClassifier

X = ['I loved this movie.', 'The product broke after a day.', 'Great service!', 'Never again.']
y = ['positive', 'negative', 'positive', 'negative']

classifier = TextClassifier(k_shot=2, random_state=0, backend='dummy').fit(X, y)
classifier.predict(['What a wonderful day.'])
```

Describe images, then classify the descriptions in one pipeline:

```python
from PIL import Image
from sklearn.pipeline import make_pipeline
from skai.generation import ImageToTextGenerator

images = [Image.new('RGB', (64, 64), color) for color in ('red', 'green', 'blue', 'white')]
pipeline = make_pipeline(
    ImageToTextGenerator(instructions='Describe the image.', backend='dummy'),
    TextClassifier(backend='dummy'),
)
pipeline.fit(images, ['warm', 'cold', 'cold', 'warm']).predict(images)
```

Make a portrait speak a text:

```python
from skai.generation import AudioToVideoGenerator, TextToAudioGenerator

portrait = Image.new('RGB', (64, 64), 'white')
pipeline = make_pipeline(
    TextToAudioGenerator(backend='dummy'),
    AudioToVideoGenerator(avatar=portrait, backend='dummy'),
)
videos = pipeline.fit(None).predict(['Hello, I am a talking portrait.'])
videos[0].duration
```

Turn a brief into a narrated, subtitled video with a recipe:

```python
from skai import recipes

explainer = recipes.explainer_video(size=(64, 48), scene_seconds=1, backend='dummy')
video = explainer.fit(None).predict(['How solar panels work'])[0]
video.duration, video.audio is not None
```

## Backends

The `backend` parameter takes `'<provider>'`, `'<provider>:<model>'` or a backend object from `skai.backends`. Without
it, components run an open-source model, locally with `hf` for most operations.

<!-- --8<-- [start:providers] -->
| Provider | Backend | Runs | Install | Tasks |
|---|---|---|---|---|
| `hf` | `HFBackend` | locally, open-source models | `scikit-ai[hf]` | every task except talking heads, voice conversion, lip sync and fine-tuning, with diarization from `scikit-ai[diarize]` |
| `openai` | `OpenAIBackend` | OpenAI API | `scikit-ai[openai]` | classification, text generation, extraction, transcription, text embeddings, object detection, moments, thumbnails, speech, image generation with reference images and edits, video generation and animation, fine-tuning |
| `litellm` | `LiteLLMBackend` | Anthropic, Gemini, Mistral, Bedrock, Azure, Ollama and others through LiteLLM | `scikit-ai[litellm]` | classification, text generation, extraction, image transcription, object detection, moments, thumbnails, text and image embeddings |
| `replicate` | `ReplicateBackend` | open-source models hosted by Replicate | `scikit-ai[replicate]` | talking heads, lip sync, voice conversion, image generation, speech synthesis with voice cloning |
| `dummy` | `DummyBackend` | locally, no model | included | every task, with placeholder outputs for tests and documentation |
<!-- --8<-- [end:providers] -->

Local operations, such as joining clips, mixing tracks, muxing, subtitles, collages, frames, shot cuts, silences and
text splitting, run without a backend. The [backends guide](https://georgedouzas.github.io/scikit-ai/guide/backends/)
lists the providers and the default model of every class and operation.

Cloud backends read credentials from environment variables, such as `OPENAI_API_KEY` or `REPLICATE_API_TOKEN`, and
never take a secret as an argument. Generated speech, converted voices, lip sync and talking heads can imitate real
people. Use them only with the consent of the people they depict, and within the policies of the providers.

## Spec-driven development

The project's engineering constitution lives in `openspec/constitution.md` and binds all work. The project uses
[OpenSpec](https://github.com/Fission-AI/OpenSpec) for spec-driven development, and `openspec/config.yaml` names the
constitution as binding.
