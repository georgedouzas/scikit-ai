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

scikit-ai provides task-oriented components for text, images, audio and video. Each component does one job, such as
describing an image, summarizing a text, transcribing speech or making a portrait speak, and follows the
[scikit-learn](https://scikit-learn.org/stable) estimator API. The same component runs on open-source models or on cloud
APIs, chosen with one `backend` parameter, and fits into `Pipeline`, `GridSearchCV` and the rest of the scikit-learn
ecosystem.

## Installation

scikit-ai is available on PyPI. The base install includes the offline `dummy` backend, and every other backend is an
optional extra:

```bash
pip install scikit-ai              # components and the dummy backend
pip install "scikit-ai[hf]"        # local open-source models with Hugging Face
pip install "scikit-ai[openai]"    # the OpenAI API
pip install "scikit-ai[litellm]"   # Anthropic, Gemini, Mistral, Bedrock, Ollama and others through LiteLLM
pip install "scikit-ai[replicate]" # open-source models hosted by Replicate
pip install "scikit-ai[video]"     # encoding and decoding video
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

Components live in packages named after their input.

| Package | Component | Does | Output |
|---|---|---|---|
| `skai.text` | `TextClassifier` | assigns labels to texts, zero-shot or few-shot | labels and probabilities |
| `skai.text` | `TextSummarizer` | summarizes texts | text |
| `skai.text` | `TextTranslator` | translates texts | text |
| `skai.text` | `TextEmbedder` | maps texts to vectors | float array |
| `skai.text` | `SpeechSynthesizer` | speaks texts, optionally in a cloned voice | audio |
| `skai.text` | `ImageGenerator` | generates images from prompts | images |
| `skai.image` | `ImageDescriptor` | describes images | text |
| `skai.image` | `ImageClassifier` | assigns labels to images, zero-shot or few-shot | labels and probabilities |
| `skai.audio` | `SpeechTranscriber` | transcribes speech, optionally with timestamps | text or timed records |
| `skai.audio` | `TalkingAvatarGenerator` | animates a portrait so that it speaks the audio | video |

## Usage

Every component is a scikit-learn estimator. The examples use the offline `dummy` backend, so they run without models
or credentials. Replace `backend='dummy'` with a backend from the table below to get real results.

Classify texts with a few labeled examples:

```python
from skai.text import TextClassifier

X = ['I loved this movie.', 'The product broke after a day.', 'Great service!', 'Never again.']
y = ['positive', 'negative', 'positive', 'negative']

classifier = TextClassifier(k_shot=2, random_state=0, backend='dummy').fit(X, y)
classifier.predict(['What a wonderful day.'])
```

Describe images, then classify the descriptions in one pipeline:

```python
from PIL import Image
from sklearn.pipeline import make_pipeline
from skai.image import ImageDescriptor

images = [Image.new('RGB', (64, 64), color) for color in ('red', 'green', 'blue', 'white')]
pipeline = make_pipeline(ImageDescriptor(backend='dummy'), TextClassifier(backend='dummy'))
pipeline.fit(images, ['warm', 'cold', 'cold', 'warm']).predict(images)
```

Make a portrait speak a text:

```python
from skai.audio import TalkingAvatarGenerator
from skai.text import SpeechSynthesizer

portrait = Image.new('RGB', (64, 64), 'white')
pipeline = make_pipeline(
    SpeechSynthesizer(backend='dummy'),
    TalkingAvatarGenerator(avatar=portrait, backend='dummy'),
)
videos = pipeline.fit(None).predict(['Hello, I am a talking portrait.'])
videos[0].duration
```

## Backends

The `backend` parameter takes `'<provider>'`, `'<provider>:<model>'` or a backend object from `skai.backends`. Without
it, components run an open-source model.

<!-- --8<-- [start:providers] -->
| Provider | Backend | Runs | Install | Tasks |
|---|---|---|---|---|
| `hf` | `HFBackend` | locally, open-source models | `scikit-ai[hf]` | every task except talking avatars |
| `openai` | `OpenAIBackend` | OpenAI API | `scikit-ai[openai]` | every task except talking avatars |
| `litellm` | `LiteLLMBackend` | Anthropic, Gemini, Mistral, Bedrock, Azure, Ollama and others through LiteLLM | `scikit-ai[litellm]` | classification, summarization, translation, image description, embeddings |
| `replicate` | `ReplicateBackend` | open-source models hosted by Replicate | `scikit-ai[replicate]` | talking avatars, image generation, speech synthesis with voice cloning |
| `dummy` | `DummyBackend` | locally, no model | included | every task, with placeholder outputs for tests and documentation |
<!-- --8<-- [end:providers] -->

The default model of every component and provider:

<!-- --8<-- [start:defaults] -->
| Component | Default backend | `hf` | `openai` | `litellm` | `replicate` |
|---|---|---|---|---|---|
| `TextClassifier` | `hf` | `facebook/bart-large-mnli` | `gpt-5-mini` | `anthropic/claude-haiku-5-5` | |
| `TextSummarizer` | `hf` | `Qwen/Qwen2.5-1.5B-Instruct` | `gpt-5-mini` | `anthropic/claude-haiku-5-5` | |
| `TextTranslator` | `hf` | `Qwen/Qwen2.5-1.5B-Instruct` | `gpt-5-mini` | `anthropic/claude-haiku-5-5` | |
| `TextEmbedder` | `hf` | `sentence-transformers/all-MiniLM-L6-v2` | `text-embedding-3-small` | `openai/text-embedding-3-small` | |
| `SpeechSynthesizer` | `hf` | `facebook/mms-tts-eng` | `gpt-4o-mini-tts` | | `jaaari/kokoro-82m`, `lucataco/xtts-v2` to clone a voice |
| `ImageGenerator` | `hf` | `stabilityai/sd-turbo` | `gpt-image-1` | | `black-forest-labs/flux-schnell` |
| `ImageDescriptor` | `hf` | `HuggingFaceTB/SmolVLM-256M-Instruct` | `gpt-5-mini` | `anthropic/claude-haiku-5-5` | |
| `ImageClassifier` | `hf` | `openai/clip-vit-base-patch32` | `gpt-5-mini` | `anthropic/claude-haiku-5-5` | |
| `SpeechTranscriber` | `hf` | `openai/whisper-small` | `whisper-1` | | |
| `TalkingAvatarGenerator` | `replicate` | | | | `cjwbw/sadtalker` |
<!-- --8<-- [end:defaults] -->

Cloud backends read credentials from environment variables, such as `OPENAI_API_KEY` or `REPLICATE_API_TOKEN`, and
never take a secret as an argument. Generated speech and talking avatars can imitate real people. Use them only with
the consent of the people they depict, and within the policies of the providers.

## Spec-driven development

The project's engineering constitution lives in `openspec/constitution.md` and binds all work. The project uses
[OpenSpec](https://github.com/Fission-AI/OpenSpec) for spec-driven development, and `openspec/config.yaml` names the
constitution as binding.
