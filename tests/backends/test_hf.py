"""Test the Hugging Face backend against fake `transformers`, `diffusers` and `sentence_transformers` modules."""

import numpy as np
import pytest
from PIL import Image

from skai.audio import SpeechTranscriber
from skai.backends import HFBackend
from skai.image import ImageClassifier, ImageDescriptor
from skai.text import ImageGenerator, SpeechSynthesizer, TextClassifier, TextEmbedder, TextSummarizer, TextTranslator
from tests.fakes import Recorder, install_module, namespace


def zero_shot(text, candidate_labels, multi_label):
    """Score the candidates in reverse order of appearance."""
    scores = np.linspace(0.6, 0.1, len(candidate_labels))
    return {'labels': list(reversed(candidate_labels)), 'scores': scores.tolist()}


PIPELINES = {
    'text-generation': lambda messages, **kwargs: [{'generated_text': 'Short summary.'}],
    'image-text-to-text': lambda text, **kwargs: [{'generated_text': 'A small image.'}],
    'zero-shot-classification': zero_shot,
    'zero-shot-image-classification': lambda image, candidate_labels: [
        {'label': label, 'score': score}
        for label, score in zip(candidate_labels, np.linspace(0.2, 0.8, len(candidate_labels)), strict=True)
    ],
    'text-to-speech': lambda text, **kwargs: {'audio': np.zeros((1, 8000)), 'sampling_rate': 16000},
    'automatic-speech-recognition': lambda inputs, **kwargs: {
        'text': ' hi there',
        'chunks': [{'timestamp': (0.0, 0.5), 'text': ' hi'}, {'timestamp': (0.5, None), 'text': ' there'}],
    },
}


@pytest.fixture
def transformers(monkeypatch):
    """Install a fake `transformers` module that records every pipeline it creates."""
    calls = {}

    def pipeline(task, model, device):
        """Create a recorded fake pipeline."""
        calls[task, model] = calls.get((task, model), 0) + 1
        return Recorder(PIPELINES[task])

    module = install_module(monkeypatch, 'transformers', pipeline=pipeline)
    module.created = calls
    return module


def test_prompted_tasks_use_chat_and_vision_models(transformers):
    """Prompt the chat model for text and the vision model for images, caching each pipeline."""
    summarizer = TextSummarizer(backend='hf').fit()
    assert summarizer.predict(['a', 'b']).tolist() == ['Short summary.'] * 2
    assert summarizer.predict(['c']).tolist() == ['Short summary.']
    assert transformers.created[('text-generation', HFBackend.default_models['summarize'])] == 1
    assert TextTranslator(target_language='French', backend='hf').fit().predict(['a'])[0] == 'Short summary.'
    assert ImageDescriptor(backend='hf').fit().predict([Image.new('RGB', (4, 4))])[0] == 'A small image.'
    assert ('image-text-to-text', HFBackend.default_models['describe_image']) in transformers.created


def test_text_classifier_uses_zero_shot_scores(transformers):
    """Return the zero-shot scores, matching descriptions back to labels."""
    classifier = TextClassifier(labels={'pos': 'positive', 'neg': 'negative'}, backend='hf').fit()
    proba = classifier.predict_proba(['great'])
    np.testing.assert_allclose(proba, [[0.1 / 0.7, 0.6 / 0.7]])
    assert classifier.predict(['great']).tolist() == ['pos']


def test_image_classifier_uses_clip_scores(transformers):
    """Return the zero-shot image scores."""
    classifier = ImageClassifier(labels=['cat', 'dog'], backend='hf').fit()
    np.testing.assert_allclose(classifier.predict_proba([Image.new('RGB', (4, 4))]), [[0.2, 0.8]])


def test_embedder_uses_sentence_transformers(transformers, monkeypatch):
    """Encode texts with a cached sentence-transformers model on the chosen device."""
    created = []

    def sentence_transformer(model, device):
        """Create a fake encoder."""
        created.append((model, device))
        return namespace(encode=lambda texts, **kwargs: np.ones((len(texts), 5)))

    install_module(monkeypatch, 'sentence_transformers', SentenceTransformer=sentence_transformer)
    embedder = TextEmbedder(dimensions=3, backend=HFBackend(device='cpu')).fit()
    np.testing.assert_allclose(embedder.transform(['a', 'b']), np.ones((2, 3)))
    assert created == [('sentence-transformers/all-MiniLM-L6-v2', 'cpu')]


def test_synthesizer_returns_mono_audio(transformers):
    """Squeeze the pipeline audio into mono samples."""
    speech = SpeechSynthesizer(backend='hf').fit().predict(['hello'])[0]
    assert speech.samples.shape == (8000,)
    assert speech.sample_rate == 16000


def test_image_generator_seeds_diffusion(transformers, monkeypatch):
    """Seed the generator and use one-step settings for the default turbo model."""
    calls = []

    class FakePipeline:
        """Record the generation arguments."""

        def to(self, device):
            """Return the pipeline on a device."""
            self.device = device
            return self

        def __call__(self, **kwargs):
            """Return an image of the requested size."""
            calls.append(kwargs)
            return namespace(images=[Image.new('RGB', (kwargs['width'], kwargs['height']))])

    install_module(
        monkeypatch,
        'diffusers',
        AutoPipelineForText2Image=namespace(from_pretrained=lambda model: FakePipeline()),
    )
    install_module(monkeypatch, 'torch', Generator=lambda: namespace(manual_seed=lambda seed: ('generator', seed)))
    generator = ImageGenerator(size=(12, 8), negative_prompt='blur', random_state=0, backend='hf').fit()
    assert generator.predict(['a cat'])[0].size == (12, 8)
    assert calls[-1]['num_inference_steps'] == 1
    assert calls[-1]['negative_prompt'] == 'blur'
    assert calls[-1]['generator'][0] == 'generator'
    ImageGenerator(size=(8, 8), backend='hf:other/model').fit().predict(['a dog'])
    assert 'num_inference_steps' not in calls[-1]


def test_transcriber_reads_chunks(transformers, audio):
    """Read the text and the timed chunks, ending an open chunk at the audio end."""
    transcriber = SpeechTranscriber(timestamps='word', language='en', backend='hf').fit()
    assert transcriber.predict([audio])[0] == [
        {'start': 0.0, 'end': 0.5, 'text': 'hi'},
        {'start': 0.5, 'end': 1.0, 'text': 'there'},
    ]
    assert SpeechTranscriber(backend='hf').fit().predict([audio]).tolist() == ['hi there']
