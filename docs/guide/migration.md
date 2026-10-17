# Migrating from 0.1

Version 0.2 refocuses scikit-ai on task-oriented components for text, images, audio and video. The `skai.llm` and
`skai.base` packages are removed.

## Text classification

`skai.llm.OpenAIClassifier` is replaced by `skai.text.TextClassifier` with the OpenAI backend:

| 0.1 | 0.2 |
|---|---|
| `OpenAIClassifier()` | `TextClassifier(backend='openai')` |
| `OpenAIClassifier(classes={0: 'negative', 1: 'positive'})` | `TextClassifier(labels={0: 'negative', 1: 'positive'}, backend='openai')` |
| `OpenAIClassifier(k_shot=3)` | `TextClassifier(k_shot=3, backend='openai')` |
| `OpenAIClassifier(openai_client=client)` | `TextClassifier(backend=OpenAIBackend(client=client))` |
| `OpenAIClassifier(responses_kwargs={...})` | `TextClassifier(backend=OpenAIBackend(options={...}))` |

Constructor parameters are keyword-only. `k_shot=None` now means zero-shot, where 0.1 selected one example per class.
The API key is read from `OPENAI_API_KEY`, or from the variable named by `OpenAIBackend(api_key_env=...)`, and `.env`
files are no longer loaded automatically.

```python exec="true" source="above" result="text"
from skai.backends import DummyBackend
from skai.text import TextClassifier

classifier = TextClassifier(labels={0: 'negative', 1: 'positive'}, backend=DummyBackend())
print(classifier.fit().predict(['I loved it.']))
```
