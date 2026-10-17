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

The API reference is in [`skai.text`](../api/text/index.md).
