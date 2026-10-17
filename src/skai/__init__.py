"""Task-oriented components for text, images, audio and video with a scikit-learn API.

Every component does one job and follows the scikit-learn estimator API, so it can be fitted, cloned, tuned and used
in pipelines. Components live in packages named after their task family, and every class name starts with its
input type, except generators, which are named for what they generate:

- `skai.classification`: assign labels to texts and images.
- `skai.description`: describe images in natural language.
- `skai.transcription`: transcribe speech.
- `skai.text`: summarize and translate texts.
- `skai.embedding`: map texts to vectors.
- `skai.generation`: generate images, speech and talking-portrait videos.

The `backend` parameter of every component chooses where it runs, for example `'hf'` for local open-source models,
`'openai'` for the OpenAI API, or `'dummy'` for offline placeholder outputs. The backends live in `skai.backends`, and
the shared media types and base classes in `skai.core`.

Examples:
    >>> from skai.text import TextSummarizer
    >>> TextSummarizer(max_length=2, backend='dummy').fit().predict(['Short texts stay short.'])[0]
    'Short texts'
"""
