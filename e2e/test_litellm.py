"""Run the LiteLLM backend through OpenAI routes, so the adapter is checked with the OpenAI key."""

import numpy as np
from sklearn.base import clone

from skai import presets
from skai.classification import AudioClassifier, TextClassifier, VideoClassifier
from skai.core import Video
from skai.embedding import TextEmbedder
from skai.extraction import TextExtractor
from skai.generation import ImageToTextGenerator, VideoToTextGenerator

CHAT = 'openai/gpt-5-mini'
DESCRIBE = 'Describe the {kind} in one sentence.'


def route(backend, model):
    """Return a copy of the backend that uses a model."""
    routed = clone(backend)
    routed.set_params(model=model)
    return routed


def test_prompted_text_tasks(litellm_backend):
    """Classify and summarize through LiteLLM."""
    backend = route(litellm_backend, CHAT)
    reviews = ['I love this, it is wonderful!', 'Terrible, it broke on the first day.']
    assert TextClassifier(labels=['positive', 'negative'], backend=backend).fit().predict(reviews).tolist() == [
        'positive',
        'negative',
    ]
    summarizer = presets.summarizer(max_length=15, backend=backend).fit()
    summary = summarizer.predict(['Solar panels turn sunlight into power.'])
    assert len(summary[0].split()) <= 15


def test_image_description(litellm_backend, shared_images):
    """Describe an image through LiteLLM."""
    descriptor = ImageToTextGenerator(instructions=DESCRIBE.format(kind='image'), backend=route(litellm_backend, CHAT))
    description = descriptor.fit().predict(shared_images[:1])[0]
    assert 'apple' in description.lower()


def test_audio_classification(litellm_backend, shared_speech):
    """Classify audio through LiteLLM with an audio model."""
    backend = route(litellm_backend, 'openai/gpt-audio-mini')
    classifier = AudioClassifier(labels=['speech', 'music', 'birdsong'], backend=backend).fit()
    assert classifier.predict([shared_speech]).tolist() == ['speech']


def test_video_tasks(litellm_backend, shared_images):
    """Classify and describe a video of the apple through LiteLLM."""
    frames = np.repeat(np.asarray(shared_images[0].convert('RGB').resize((256, 256)))[np.newaxis], 10, axis=0)
    video = Video(frames, fps=5)
    backend = route(litellm_backend, CHAT)
    classifier = VideoClassifier(labels=['fruit', 'vehicle'], n_frames=2, backend=backend).fit()
    assert classifier.predict([video]).tolist() == ['fruit']
    descriptor = VideoToTextGenerator(instructions=DESCRIBE.format(kind='video'), n_frames=2, backend=backend)
    assert 'apple' in descriptor.fit().predict([video])[0].lower()


def test_extraction(litellm_backend):
    """Extract fields with JSON mode through LiteLLM."""
    extractor = TextExtractor(fields={'name': 'person name', 'city': 'city'}, backend=route(litellm_backend, CHAT))
    record = extractor.fit().predict(['Ada Lovelace lived in London.'])[0]
    assert 'Ada' in str(record['name'])
    assert 'London' in str(record['city'])


def test_text_embeddings(litellm_backend):
    """Embed and natively shorten texts through LiteLLM."""
    embedder = TextEmbedder(dimensions=8, backend=route(litellm_backend, 'openai/text-embedding-3-small'))
    assert embedder.fit(['a', 'b', 'c']).transform(['a cat']).shape == (1, 8)
