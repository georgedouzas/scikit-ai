"""Run every component that the OpenAI backend supports on real requests."""

import re

import numpy as np
import pytest
from sklearn.pipeline import make_pipeline

from skai.classification import ImageClassifier, TextClassifier
from skai.core import Audio
from skai.description import ImageDescriptor
from skai.embedding import TextEmbedder
from skai.generation import ImageGenerator, SpeechSynthesizer
from skai.text import TextSummarizer, TextTranslator
from skai.transcription import SpeechTranscriber

ARTICLE = (
    'Solar panels turn sunlight into electricity. Homes with panels on the roof can cut their power bills, and any '
    'extra energy can be sold back to the grid. Panels last for about twenty-five years and need little care, but '
    'they produce less power on cloudy days and nothing at night, so many owners add a battery.'
)
GREEK = re.compile('[Ͱ-Ͽ]')


def test_summarizer_and_translator(openai_backend):
    """Summarize within the word limit and translate into Greek script."""
    summary = TextSummarizer(max_length=25, backend=openai_backend).fit().predict([ARTICLE])[0]
    assert len(summary.split()) <= 25
    assert 'solar' in summary.lower()
    translation = TextTranslator(target_language='Greek', backend=openai_backend).fit().predict(['Good morning'])[0]
    assert GREEK.search(translation)


def test_text_classifier_zero_and_few_shot(openai_backend):
    """Classify clear-cut reviews zero-shot and few-shot."""
    reviews = ['Absolutely wonderful, I love it!', 'Terrible. It broke on day one and support ignored me.']
    zero_shot = TextClassifier(labels=['positive', 'negative'], backend=openai_backend).fit()
    assert zero_shot.predict(reviews).tolist() == ['positive', 'negative']
    few_shot = TextClassifier(k_shot=2, random_state=0, backend=openai_backend)
    few_shot.fit(['Great value.', 'Awful quality.'], ['positive', 'negative'])
    assert few_shot.predict(reviews).tolist() == ['positive', 'negative']
    np.testing.assert_allclose(few_shot.predict_proba(reviews).sum(axis=1), 1)


def test_speech_round_trip(openai_backend):
    """Keep the key words of a sentence through speech synthesis and transcription."""
    sentence = 'The quick brown fox jumps over the lazy dog.'
    speech = SpeechSynthesizer(backend=openai_backend).fit().predict([sentence])
    assert isinstance(speech[0], Audio)
    assert speech[0].duration > 1
    transcript = SpeechTranscriber(language='en', backend=openai_backend).fit().predict(speech)[0].lower()
    assert 'fox' in transcript
    assert 'dog' in transcript
    words = SpeechTranscriber(timestamps='word', backend=openai_backend).fit().predict(speech)[0]
    assert words
    assert all(record['start'] <= record['end'] for record in words)


def test_spoken_reviews_pipeline(openai_backend):
    """Classify spoken reviews through a transcriber and classifier pipeline."""
    reviews = ['I am thrilled with this purchase.', 'This was a complete waste of money.']
    speech = SpeechSynthesizer(backend=openai_backend).fit().predict(reviews)
    pipeline = make_pipeline(
        SpeechTranscriber(backend=openai_backend),
        TextClassifier(labels=['positive', 'negative'], backend=openai_backend),
    )
    assert pipeline.fit(speech).predict(speech).tolist() == ['positive', 'negative']


def test_image_generation_and_understanding(openai_backend):
    """Generate an image, then describe and classify it."""
    generator = ImageGenerator(size=(1024, 1024), backend=openai_backend).fit()
    image = generator.predict(['A single red apple on a white table'])[0]
    assert image.size == (1024, 1024)
    description = ImageDescriptor(backend=openai_backend).fit().predict([image])[0]
    assert 'apple' in description.lower()
    classifier = ImageClassifier(labels=['apple', 'car', 'dog'], backend=openai_backend).fit()
    assert classifier.predict([image]).tolist() == ['apple']


def test_embeddings_rank_paraphrases_first(openai_backend):
    """Place a paraphrase closer than an unrelated sentence."""
    texts = ['The cat sat on the mat.', 'A cat was sitting on the rug.', 'Interest rates rose by half a percent.']
    vectors = TextEmbedder(normalize=True, backend=openai_backend).fit().transform(texts)
    similarities = vectors @ vectors[0]
    assert similarities[1] > similarities[2]


def test_video_generation(video_enabled):
    """Generate a short video, once `VideoGenerator` exists."""
    if not video_enabled:
        pytest.skip('Set SKAI_E2E_VIDEO=1 to run the video generation tests.')
    pytest.skip('VideoGenerator is implemented in group 6 of add-roadmap-components.')
