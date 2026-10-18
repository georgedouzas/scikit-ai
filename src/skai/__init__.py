"""Components for text, images, audio and video with a scikit-learn API, named by input and output.

Every component follows the scikit-learn estimator API, so it can be fitted, cloned, tuned and used in pipelines.
Analyzers are named `{Input}{Role}` and generators `{Input}To{Output}Generator`, where the input is `Text`, `Image`,
`Audio`, `Video` or `Multimodal`:

- `skai.classification`: assign labels.
- `skai.extraction`: extract structured records.
- `skai.embedding`: map samples to vectors.
- `skai.segmentation`: split samples into regions in space or time.
- `skai.generation`: generate texts, images, audio and videos.
- `skai.pipeline`: combine branches, pick fields, and work on every item of split samples.
- `skai.presets` and `skai.recipes`: configured components and pipelines for common tasks.

The `backend` parameter of every component chooses where it runs, for example `'hf'` for local open-source models,
`'openai'` for the OpenAI API, or `'dummy'` for offline placeholder outputs. The backends live in `skai.backends`, and
the shared media types and base classes in `skai.core`.

Examples:
    >>> from skai.generation import TextToTextGenerator
    >>> generator = TextToTextGenerator(instructions='Summarize the text.', max_length=2, backend='dummy')
    >>> generator.fit().predict(['Short texts stay short.'])[0]
    'Short texts'
"""
