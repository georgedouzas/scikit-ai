# Text

`skai.text` holds the components that turn texts into other texts. `TextSummarizer` returns summaries of at most
`max_length` words in an optional `style`. `TextTranslator` translates into `target_language`, from `source_language`
or from the detected language.

```python exec="true" source="above" result="text"
from skai.text import TextSummarizer, TextTranslator

text = 'scikit-ai gives every media task the same scikit-learn interface, whatever model runs it.'
print(TextSummarizer(max_length=6, style='one line', backend='dummy').fit().predict([text]))
print(TextTranslator(target_language='Greek', backend='dummy').fit().predict(['Good morning']))
```

## What fit learns

`fit(X, y)` with reference outputs as `y` passes text and reference pairs to the backend. `TextSummarizer` also
learns the summary length when `max_length=None`, and `TextTranslator` learns a glossary of terms that recur in the
texts, so they are translated consistently.

```python exec="true" source="above" result="text"
from skai.text import TextSummarizer, TextTranslator

summarizer = TextSummarizer(max_length=None, backend='dummy').fit(['one two three four'], ['one two'])
translator = TextTranslator(target_language='French', backend='dummy')
translator.fit(['Open SkaiBox.', 'SkaiBox is ready.'], ['Ouvrez SkaiBox.', 'SkaiBox est prêt.'])
print(summarizer.max_length_, translator.glossary_)
```

The API reference is in [`skai.text`](../api/text/index.md).
