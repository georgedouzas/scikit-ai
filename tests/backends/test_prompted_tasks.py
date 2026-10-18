"""Test the prompted tasks of the OpenAI and LiteLLM backends against fake clients."""

import json

import numpy as np
import pytest
from PIL import Image

from skai.backends import LiteLLMBackend, OpenAIBackend
from skai.classification import MultimodalClassifier
from skai.core import Video
from skai.extraction import AudioExtractor, ImageExtractor, TextExtractor, VideoExtractor
from skai.generation import (
    AudioToTextGenerator,
    ImageToTextGenerator,
    MultimodalToTextGenerator,
    TextToTextGenerator,
    VideoToImageGenerator,
    VideoToTextGenerator,
)
from skai.segmentation import AudioSegmenter, ImageSegmenter, VideoSegmenter
from tests.fakes import Recorder, chat_response, install_module, namespace

OPENAI_CHAT = 'gpt-5-mini'
OPENAI_AUDIO = 'gpt-audio-mini'
LITELLM_CHAT = 'anthropic/claude-haiku-5-5'
LITELLM_MEDIA = 'gemini/gemini-2.5-flash'


class Provider:
    """Hold a backend whose fake client answers with a settable text and records the model of every call."""

    def __init__(self, name, monkeypatch):
        """Create the fake client of a provider."""
        self.name = name
        self.answer = 'ok'
        create = Recorder(lambda **kwargs: chat_response(self.answer))
        self.calls = create.calls
        if name == 'openai':
            client = namespace(chat=namespace(completions=namespace(create=create)))
            self.backend = OpenAIBackend(client=client)
            self.chat, self.audio, self.video = OPENAI_CHAT, OPENAI_AUDIO, OPENAI_CHAT
        else:
            install_module(monkeypatch, 'litellm', completion=create, embedding=Recorder(lambda **kwargs: None))
            self.backend = LiteLLMBackend()
            self.chat, self.audio, self.video = LITELLM_CHAT, LITELLM_MEDIA, LITELLM_MEDIA

    @property
    def model(self):
        """The model of the last call."""
        return self.calls[-1]['model']

    @property
    def parts(self):
        """The types of the content parts of the last user message."""
        return [part['type'] for part in self.calls[-1]['messages'][1]['content']]


@pytest.fixture(params=['openai', 'litellm'])
def provider(request, monkeypatch):
    """Return each prompted provider with a fake client."""
    return Provider(request.param, monkeypatch)


@pytest.fixture
def video(audio):
    """Return a short video with an audio track."""
    return Video(np.zeros((8, 16, 16, 3), dtype=np.uint8), fps=4, audio=audio)


def test_text_generation_per_input_kind(provider, audio, video):
    """Send every input kind to the model of its kind and return the stripped answer."""
    provider.answer = ' result '
    image = Image.new('RGB', (8, 8))
    cases = [
        (TextToTextGenerator(instructions='Summarize.'), 'text', provider.chat),
        (ImageToTextGenerator(instructions='Describe.'), image, provider.chat),
        (AudioToTextGenerator(operation='instruct', instructions='Describe.'), audio, provider.audio),
        (VideoToTextGenerator(instructions='Describe.', n_frames=2), video, provider.video),
        (MultimodalToTextGenerator(instructions='Caption.'), {'photo': image, 'notes': 'Crete'}, provider.chat),
        (MultimodalToTextGenerator(instructions='Caption.'), {'clip': audio, 'notes': 'Crete'}, provider.audio),
    ]
    for generator, sample, model in cases:
        assert generator.set_params(backend=provider.backend).fit().predict([sample]).tolist() == ['result']
        assert provider.model == model


def test_multimodal_parts_are_named(provider):
    """Name every part of a multimodal sample by its key and send its image."""
    provider.answer = 'safe'
    classifier = MultimodalClassifier(labels=['safe', 'unsafe'], backend=provider.backend).fit()
    assert classifier.predict([{'image': Image.new('RGB', (8, 8)), 'caption': 'hello'}]).tolist() == ['safe']
    text = provider.calls[-1]['messages'][1]['content'][0]['text']
    assert 'image: <image 1>' in text
    assert 'caption: hello' in text
    assert provider.parts == ['text', 'image_url']


def test_extraction_per_input_kind(provider, audio, video):
    """Ask for JSON from every input kind and keep exactly the declared fields."""
    provider.answer = json.dumps({'topic': 'bees', 'extra': 1})
    cases = [
        (TextExtractor, 'text', provider.chat),
        (ImageExtractor, Image.new('RGB', (8, 8)), provider.chat),
        (AudioExtractor, audio, provider.audio),
        (VideoExtractor, video, provider.video),
    ]
    for extractor_class, sample, model in cases:
        extractor = extractor_class(fields={'topic': 'main topic'}, backend=provider.backend).fit()
        assert extractor.predict([sample])[0] == {'topic': 'bees'}
        assert provider.model == model


def test_image_transcription(provider):
    """Transcribe the text of an image with the vision model, asking for Markdown when requested."""
    provider.answer = '# Title'
    generator = ImageToTextGenerator(operation='transcribe', format='markdown', backend=provider.backend).fit()
    assert generator.predict([Image.new('RGB', (8, 8))]).tolist() == ['# Title']
    assert 'Markdown' in provider.calls[-1]['messages'][1]['content'][0]['text']
    assert provider.model == provider.chat


def test_object_detection(provider):
    """Read the boxes of the answer, clamped to the image, and crop the regions."""
    provider.answer = json.dumps({'regions': [{'label': 'cat', 'box': [2, 2, 40, 6], 'score': 0.8}]})
    segmenter = ImageSegmenter(labels=['cat'], backend=provider.backend).fit()
    (region,) = segmenter.predict([Image.new('RGB', (16, 8))])[0]
    assert region['box'] == (2, 2, 16, 6)
    assert region['media'].size == (14, 4)


def test_moments_in_audio_and_video(provider, audio, video):
    """Find moments of audio with the audio model and of videos with the video model."""
    provider.answer = json.dumps({'moments': [{'start': 0.2, 'end': 0.6, 'score': 0.9}]})
    (moment,) = AudioSegmenter(by='moment', prompt='a bark', backend=provider.backend).fit().predict([audio])[0]
    assert (moment['start'], moment['end']) == (0.2, 0.6)
    assert provider.model == provider.audio
    VideoSegmenter(by='moment', prompt='a goal', n_frames=2, backend=provider.backend).fit().predict([video])
    assert provider.model == provider.video


def test_thumbnail(provider, video):
    """Return the frame the model names."""
    provider.answer = 'Frame 2.'
    thumbnail = VideoToImageGenerator(n_frames=2, backend=provider.backend).fit().predict([video])[0]
    assert thumbnail.size == (16, 16)
    assert provider.parts == ['text', 'image_url', 'image_url']
