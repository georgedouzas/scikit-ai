"""Run the Hugging Face backend with small local models, behind `SKAI_E2E_HF=1` because it downloads them."""

import numpy as np
import pytest
from PIL import Image

from skai.classification import AudioClassifier, ImageClassifier, TextClassifier
from skai.core import Audio
from skai.description import ImageDescriptor
from skai.embedding import AudioEmbedder, ImageEmbedder, TextEmbedder
from skai.generation import ImageGenerator, SpeechSynthesizer
from skai.text import TextExtractor, TextSummarizer, TextTranslator
from skai.transcription import SpeechTranscriber

GREEK = 'Ͱ-Ͽ'


@pytest.fixture(autouse=True)
def require_hf(hf_enabled):
    """Skip unless the Hugging Face tests were requested."""
    if not hf_enabled:
        pytest.skip('Set SKAI_E2E_HF=1 to run the Hugging Face tests, which download models.')


@pytest.fixture(scope='module')
def red_image():
    """Return a picture of a red apple drawn with Pillow."""
    image = Image.new('RGB', (224, 224), 'white')
    for x in range(224):
        for y in range(224):
            if (x - 112) ** 2 + (y - 120) ** 2 < 70**2:
                image.putpixel((x, y), (200, 20, 20))
    return image


def test_zero_shot_text_classification():
    """Classify clear-cut reviews with the natural language inference model."""
    classifier = TextClassifier(labels=['positive', 'negative'], backend='hf').fit()
    assert classifier.predict(['I love it, it is wonderful.', 'I hate it, it is awful.']).tolist() == [
        'positive',
        'negative',
    ]


def test_zero_shot_image_classification(red_image):
    """Classify a red disc as red with CLIP."""
    classifier = ImageClassifier(labels=['a red object', 'a blue object'], backend='hf').fit()
    assert classifier.predict([red_image]).tolist() == ['a red object']


def test_text_and_image_embeddings(red_image):
    """Rank a paraphrase first with MiniLM and embed an image with CLIP."""
    texts = ['The cat sat on the mat.', 'A cat was sitting on a rug.', 'Stocks fell sharply today.']
    vectors = TextEmbedder(normalize=True, center=False, backend='hf').fit().transform(texts)
    assert (vectors @ vectors[0])[1] > (vectors @ vectors[0])[2]
    assert ImageEmbedder(backend='hf').fit().transform([red_image]).shape == (1, 512)


def test_speech_round_trip():
    """Speak a sentence with MMS-TTS and transcribe it with Whisper."""
    speech = SpeechSynthesizer(backend='hf').fit().predict(['The weather is nice today.'])
    assert isinstance(speech[0], Audio)
    transcript = SpeechTranscriber(language='en', backend='hf').fit().predict(speech)[0].lower()
    assert 'weather' in transcript


def test_audio_classification_and_embeddings():
    """Classify a pure tone and embed it with CLAP."""
    times = np.arange(48000) / 48000
    tone = Audio((0.3 * np.sin(2 * np.pi * 440 * times)).astype(np.float32), 48000)
    classifier = AudioClassifier(labels=['a beep tone', 'a dog barking'], backend='hf').fit()
    assert classifier.predict([tone]).tolist() == ['a beep tone']
    assert AudioEmbedder(backend='hf').fit().transform([tone]).shape[0] == 1


def test_image_description(red_image):
    """Describe the red disc with SmolVLM."""
    description = ImageDescriptor(backend='hf').fit().predict([red_image])[0].lower()
    assert 'red' in description


def test_prompted_text_tasks():
    """Summarize, translate and extract with Qwen2.5-1.5B."""
    article = 'The city opened a new library with ten thousand books and free study rooms for students.'
    assert 'library' in TextSummarizer(max_length=20, backend='hf').fit().predict([article])[0].lower()
    translation = TextTranslator(target_language='Greek', backend='hf').fit().predict(['Good morning, friend.'])[0]
    assert any('Ͱ' <= character <= 'Ͽ' for character in translation)
    record = TextExtractor(fields={'name': 'person name', 'city': 'city'}, backend='hf').fit()
    assert 'Ada' in str(record.predict(['Ada lives in London.'])[0]['name'])


def test_image_generation():
    """Generate an image of the requested size with SD-Turbo."""
    image = ImageGenerator(size=(512, 512), random_state=0, backend='hf').fit().predict(['a red apple'])[0]
    assert image.size == (512, 512)
