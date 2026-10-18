# Backends

The `backend` parameter of every component chooses where the component runs. The interface of the component stays the
same whatever the backend.

## Choosing a backend

`backend` takes a string or a backend object:

- `'<provider>'` uses the default model of the provider for the task, for example `'openai'`.
- `'<provider>:<model>'` uses a specific model, for example `'openai:gpt-5-mini'` or
  `'hf:HuggingFaceTB/SmolVLM-256M-Instruct'`.
- A backend object from `skai.backends` holds the model and the provider settings, such as the credential variable,
  a preconfigured client, an endpoint and extra call options.
- `None` uses the default backend of the component and operation, which runs an open-source model: `hf` locally for
  most operations, and open-source models hosted by `replicate` for the others.

```python exec="true" source="above" result="text"
from skai.backends import DummyBackend
from skai.generation import TextToTextGenerator

instructions = 'Translate into Greek.'
by_string = TextToTextGenerator(instructions=instructions, backend='dummy:any-model').fit()
by_object = TextToTextGenerator(instructions=instructions, backend=DummyBackend(model='any-model')).fit()
print(by_string.backend_, by_object.backend_)
```

Backend objects are scikit-learn estimators, so their parameters can be tuned like any nested parameter:

```python exec="true" source="above" result="text"
from sklearn.model_selection import GridSearchCV
from skai.backends import DummyBackend
from skai.classification import TextClassifier

X = ['good', 'bad', 'great', 'awful'] * 3
y = ['positive', 'negative'] * 6
search = GridSearchCV(TextClassifier(backend=DummyBackend()), {'backend__model': ['model-a', 'model-b']}, cv=2)
print(search.fit(X, y).best_params_)
```

## Providers

--8<-- "README.md:providers"

A component raises `UnsupportedBackendError`, a `ValueError`, at `fit` when its provider is unknown or does not support
its task on its input kind, and the message lists the providers that do. A missing extra raises `MissingExtraError`,
an `ImportError`, that names the `pip install` command.

## Tasks

Every operation of a component runs one backend task, and local operations run no task. The backend tasks are:

- `classify_text`, `classify_image`, `classify_audio`, `classify_video` and `classify_multimodal` for classifiers.
- `generate_text` for every text generator with `operation='instruct'`. It replaces the `summarize`, `translate` and
  `describe_*` tasks of 0.2.
- `extract_fields` for every extractor, whose request carries the input kind.
- `transcribe` for speech, `transcribe_image` for the text written in images.
- `embed_text`, `embed_image` and `embed_audio` for embedders. `VideoEmbedder` averages `embed_image` over frames.
- `detect_objects`, `segment_image`, `diarize` and `find_moments` for segmenters and reframing.
- `synthesize_speech`, `generate_music`, `generate_sound`, `separate_audio`, `enhance_audio` and `convert_voice` for
  audio.
- `generate_image`, `edit_image`, `upscale_image`, `remove_background` and `pick_thumbnail` for images.
- `generate_video`, `animate_image`, `animate_avatar` and `lip_sync` for video.
- `fine_tune` for `adaptation='finetune'`.

`Backend.supported_tasks()` returns the tasks a backend class implements, and `Backend.supports(task, input_kind)`
says whether it runs a task on samples of an input kind. A prompted backend runs `generate_text` on audio only when
its models accept audio, for example. `skai.core.list_backends` lists the registered providers that support a task.

```python exec="true" source="above" result="text"
from skai.backends import HFBackend, OpenAIBackend
from skai.core import list_backends

print(OpenAIBackend.supports('generate_text', 'audio'), HFBackend.supports('extract_fields', 'audio'))
print(list_backends('generate_text'), list_backends('animate_avatar'))
```

## Providers and default models of every operation

A key of `default_models` is a task, or `'<task>:<kind>'` for a default that depends on the input kind, such as
`'generate_text:image'` for a vision model. The table lists, for every class and operation, the providers that support
it and their default model, or "set `model`" where the provider has no default for it. Operations that no cloud API
offers are marked as open-source only, and run locally on `hf` or on open-source models hosted by `replicate`.
Local operations need no backend. The default column names the backend used when `backend` is `None`.

```python exec="true"
from skai import classification, embedding, extraction, generation, segmentation
from skai.backends import HFBackend, LiteLLMBackend, OpenAIBackend, ReplicateBackend
from skai.core import UnsupportedBackendError

PROVIDERS = {'hf': HFBackend, 'openai': OpenAIBackend, 'litellm': LiteLLMBackend, 'replicate': ReplicateBackend}
CLOUD = {'openai', 'litellm'}


def describe(backend, task, kind):
    try:
        return f'`{backend().resolve_model(task, kind)}`'
    except UnsupportedBackendError:
        return 'set `model`'


print('| Class | `operation` | Task | Default | ' + ' | '.join(f'`{name}`' for name in PROVIDERS) + ' | Notes |')
print('|---|---|---|---|' + '---|' * len(PROVIDERS) + '---|')
for module in (classification, extraction, embedding, segmentation, generation):
    for name in module.__all__:
        component = getattr(module, name)
        for operation, task in component.operations.items():
            if task is None:
                print(f'| `{name}` | `{operation}` | local | |' + ' |' * len(PROVIDERS) + ' no backend |')
                continue
            cells = {
                provider: describe(backend, task, component.input_kind)
                for provider, backend in PROVIDERS.items()
                if backend.supports(task, component.input_kind)
            }
            note = '' if CLOUD & set(cells) else 'open-source only'
            if not cells:
                note += ', no bundled provider yet'
            row = ' | '.join(cells.get(provider, '') for provider in PROVIDERS)
            default = component.default_backends.get(operation, component.default_backend)
            print(f'| `{name}` | `{operation}` | `{task}` | `{default}` | {row} | {note} |')
```

Local Hugging Face models are downloaded on first use. Replicate models should be pinned to a version, as in
`'replicate:cjwbw/sadtalker:<version>'`, for reproducible results. `ReplicateBackend` clones voices with
`lucataco/xtts-v2` when `voice` is reference audio.

Prompted backends (`openai`, `litellm`, and `hf` for vision and chat models) score classification with one for every
label the model names and zero for the others. The `hf` backend classifies texts, images and audio with zero-shot
models that return real scores, and ignores few-shot examples. `dummy` returns deterministic placeholder outputs, so
the same request always gives the same result.

## Credentials

Cloud backends never take a secret as an argument. They take the name of the environment variable that holds it, and
read it when the client is created in `fit`:

| Backend | Parameter | Default variable |
|---|---|---|
| `OpenAIBackend` | `api_key_env` | `OPENAI_API_KEY` |
| `ReplicateBackend` | `api_token_env` | `REPLICATE_API_TOKEN` |
| `LiteLLMBackend` | `api_key_env` | the standard variable of each provider, such as `ANTHROPIC_API_KEY` |

A missing variable raises `MissingCredentialError` naming the variable. `OpenAIBackend` and `ReplicateBackend` also
take a preconfigured `client`, which clones of the backend share. Clients are not pickled, and are created again when
a restored component is used.

## Registering a backend

A backend subclasses `skai.core.Backend` and overrides the methods of the tasks it supports, such as `generate_text` or
`generate_image`. Every method takes one request from `skai.core`, such as `TextGenerationRequest`, and returns one
output. A subclass of `skai.core.PromptedBackend` implements `complete` and gets classification, text generation,
extraction, image transcription, object detection, moment finding and thumbnail picking from it.
`skai.core.register` makes the backend available by name.

```python exec="true" source="above" result="text"
from skai.core import Backend, TextGenerationRequest, register
from skai.generation import TextToTextGenerator


class FirstSentenceBackend(Backend):
    def generate_text(self, request: TextGenerationRequest) -> str:
        return request.sample.split('.')[0] + '.'


register('first-sentence', FirstSentenceBackend)
generator = TextToTextGenerator(instructions='Summarize the text.', backend='first-sentence').fit()
print(generator.predict(['It works. Everything else is detail.']))
```

Packages can also declare backends as entry points in the `skai.backends` group, which scikit-ai loads on first use:

```toml
[project.entry-points."skai.backends"]
my-provider = "my_package:MyBackend"
```
