# Presets

`skai.presets` gives common tasks a name. Every preset is a function that returns an ordinary component with its
`instructions`, `operation` or `by` set, and passes extra keyword arguments, such as `backend`, `max_length` or
`k_shot`, to the component. The result clones, tunes with `GridSearchCV` and fits in a pipeline exactly like the
component it is.

```python exec="true" source="above" result="text"
from sklearn.base import clone
from skai import presets

summarizer = presets.summarizer(max_length=50, backend='dummy')
print(type(summarizer).__name__, summarizer.max_length, summarizer.instructions)
print(clone(presets.translator('Greek', backend='dummy')).fit().predict(['Good morning']))
```

## Available presets

The table lists every preset, the component it returns, and the parameters it sets besides the arguments shown,
which are required.

```python exec="true"
from skai import presets

ARGUMENTS = {
    'object_detector': {'labels': ['person']},
    'question_answerer': {'question': 'What is shown?'},
    'talking_head': {'avatar': 'portrait.png'},
    'translator': {'target_language': 'Greek'},
    'voice_converter': {'voice': 'voice.wav'},
}

print('| Preset | Returns | Configuration |')
print('|---|---|---|')
for name in presets.__all__:
    if name == 'INSTRUCTIONS':
        continue
    arguments = ARGUMENTS.get(name, {})
    component = getattr(presets, name)(**arguments)
    call = ', '.join(f'{key}={value!r}' for key, value in arguments.items())
    defaults = type(component)().get_params()
    settings = {
        key: value
        for key, value in component.get_params().items()
        if key in ('operation', 'by') or (value != defaults[key] and key not in arguments)
    }
    configuration = '<br>'.join(f'`{key}={value!r}`' for key, value in settings.items())
    print(f'| `presets.{name}({call})` | `{type(component).__name__}` | {configuration} |')
```

## Instructions

`skai.presets.INSTRUCTIONS` holds the instructions of the presets that are text generators, extractors or
classifiers. The names in braces are filled from the arguments of the preset, and `{kind}` is the input kind, such
as `image` or `video`. The strings can be reused or adapted in a component of your own.

```python exec="true"
from skai.presets import INSTRUCTIONS

print('| Preset | Instructions |')
print('|---|---|')
for name, instructions in INSTRUCTIONS.items():
    print(f'| `{name}` | {instructions} |')
```

```python exec="true" source="above" result="text"
from skai.generation import VideoToTextGenerator
from skai.presets import INSTRUCTIONS

instructions = INSTRUCTIONS['question_answerer'].format(kind='video', question='Who scores?')
print(VideoToTextGenerator(instructions=instructions, backend='dummy').instructions)
```

The API reference is in [`skai.presets`](../api/presets/index.md).
