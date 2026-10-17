"""Check on the OpenAI backend that every component learns from the data given to `fit`."""

import re

import numpy as np
import pytest

from skai.classification import TextClassifier
from skai.core import FineTuneError
from skai.description import ImageDescriptor
from skai.embedding import TextEmbedder
from skai.generation import SpeechSynthesizer
from skai.text import TextSummarizer, TextTranslator
from skai.transcription import SpeechTranscriber

GREEK = re.compile('[Ͱ-Ͽ]')
ARTICLES = [
    'The city opened a new library with ten thousand books, a café and free study rooms for students.',
    'A local bakery won the national bread contest after only two years in business.',
    'Heavy rain closed the main road for six hours, and the council promised new drains.',
]
TLDR_SUMMARIES = [
    'TL;DR: New library with books, café and study rooms.',
    'TL;DR: Young bakery wins national bread contest.',
    'TL;DR: Rain closed the main road; new drains promised.',
]


def test_summarizer_follows_reference_style(openai_backend):
    """Copy the format and the length of the reference summaries."""
    summarizer = TextSummarizer(max_length=None, backend=openai_backend).fit(ARTICLES, TLDR_SUMMARIES)
    assert summarizer.max_length_ <= 10
    summary = summarizer.predict(['The museum extended its opening hours to ten at night during the summer.'])[0]
    assert summary.startswith('TL;DR')


def test_translator_keeps_glossary_terms(openai_backend):
    """Keep a product name that the references never translate."""
    sources = ['Open SkaiBox to start.', 'SkaiBox saves your work.', 'Thank you.']
    references = ['Ανοίξτε το SkaiBox για να ξεκινήσετε.', 'Το SkaiBox αποθηκεύει τη δουλειά σας.', 'Ευχαριστώ.']
    translator = TextTranslator(target_language='Greek', backend=openai_backend).fit(sources, references)
    assert translator.glossary_ == (('SkaiBox', 'SkaiBox'),)
    translation = translator.predict(['SkaiBox is closing now.'])[0]
    assert 'SkaiBox' in translation
    assert GREEK.search(translation)


def test_descriptor_follows_reference_style(openai_backend, shared_images):
    """Describe a new image in the format of the reference descriptions."""
    apple, car = shared_images
    descriptor = ImageDescriptor(backend=openai_backend).fit([apple], ['Object: apple. Color: red.'])
    description = descriptor.predict([car])[0]
    assert 'Object:' in description
    assert 'Color:' in description


def test_transcriber_spells_learned_terms(openai_backend):
    """Spell a made-up product name as the reference transcripts do."""
    speech = SpeechSynthesizer(backend=openai_backend).fit().predict(['Please restart the Skaivox server tonight.'])
    references = ['The Skaivox server is down.', 'Skaivox needs an update.']
    transcriber = SpeechTranscriber(language='en', backend=openai_backend).fit([speech[0], speech[0]], references)
    assert 'Skaivox' in transcriber.vocabulary_
    assert 'Skaivox' in transcriber.predict(speech)[0]


def test_embedder_centers_and_shortens(openai_backend):
    """Center the embeddings of the training texts and shorten them natively."""
    corpus = ARTICLES + TLDR_SUMMARIES
    embedder = TextEmbedder(dimensions=8, backend=openai_backend).fit(corpus)
    assert embedder.components_ is None
    embeddings = embedder.transform(corpus)
    assert embeddings.shape == (6, 8)
    raw = TextEmbedder(dimensions=8, center=False, backend=openai_backend).fit(corpus).transform(corpus)
    assert np.abs(raw.mean(axis=0)).max() > 1e-2
    np.testing.assert_allclose(embeddings.mean(axis=0), 0, atol=1e-3)


def test_classifier_fine_tuning(openai_backend, finetune_enabled):
    """Fine-tune a classifier and predict with the new model."""
    if not finetune_enabled:
        pytest.skip('Set SKAI_E2E_FINETUNE=1 to run the fine-tuning tests.')
    texts = [f'I {word} this product.' for word in ['love', 'adore', 'like', 'enjoy', 'cherish']]
    texts += [f'I {word} this product.' for word in ['hate', 'loathe', 'dislike', 'despise', 'detest']]
    labels = ['positive'] * 5 + ['negative'] * 5
    classifier = TextClassifier(adaptation='finetune', backend=openai_backend)
    try:
        classifier.fit(texts, labels)
    except FineTuneError as error:
        if 'refused' not in str(error):
            raise
        pytest.skip(f'The OpenAI account cannot fine-tune: {error}')
    assert classifier.model_.startswith('ft:')
    assert classifier.predict(['I really love it.', 'I really hate it.']).tolist() == ['positive', 'negative']
