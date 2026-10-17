"""Task-oriented components for text, images, audio and video with a scikit-learn API.

Every component does one job and follows the scikit-learn estimator API, so it can be fitted, cloned, tuned and used
in pipelines. Components live in packages named after their input:

- `skai.text`: classify, summarize, translate and embed texts, speak them and generate images from them.
- `skai.image`: describe and classify images.
- `skai.audio`: transcribe speech and animate a portrait that speaks it.

The `backend` parameter of every component chooses where it runs, for example `'hf'` for local open-source models,
`'openai'` for the OpenAI API, or `'dummy'` for offline placeholder outputs. The backends live in `skai.backends`, and
the shared media types and base classes in `skai.core`.

Examples:
    >>> from skai.text import TextSummarizer
    >>> TextSummarizer(max_length=2, backend='dummy').fit().predict(['Short texts stay short.'])[0]
    'Short texts'
"""
