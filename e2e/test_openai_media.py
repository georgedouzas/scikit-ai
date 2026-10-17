"""Run the audio, video, extraction and editing components on the OpenAI backend."""

import numpy as np
import pytest

from skai.classification import AudioClassifier, VideoClassifier
from skai.core import Video
from skai.description import AudioDescriptor, ImageDescriptor, VideoDescriptor
from skai.generation import ImageAnimator, ImageEditor, SpeechSynthesizer, VideoGenerator
from skai.text import TextExtractor
from skai.transcription import VideoTranscriber

SPEECH_WORDS = ('speech', 'speak', 'voice', 'talk', 'say', 'person', 'man', 'woman')


@pytest.fixture(scope='module')
def speech(request):
    """Synthesize one sentence of speech once for the module."""
    backend = request.getfixturevalue('openai_backend')
    return SpeechSynthesizer(backend=backend).fit().predict(['Good morning, the apple harvest starts today.'])[0]


@pytest.fixture(scope='module')
def apple_video(shared_images, speech):
    """Build a video of the generated apple image that speaks the synthesized sentence."""
    frames = np.repeat(np.asarray(shared_images[0].convert('RGB').resize((320, 320)))[np.newaxis], 25, axis=0)
    return Video(frames, fps=25 / speech.duration, audio=speech)


def test_audio_classifier_recognizes_speech(openai_backend, speech):
    """Label synthesized speech as speech."""
    classifier = AudioClassifier(labels=['speech', 'music', 'birdsong'], backend=openai_backend).fit()
    assert classifier.predict([speech]).tolist() == ['speech']


def test_audio_descriptor_mentions_speech(openai_backend, speech):
    """Describe synthesized speech as someone speaking."""
    description = AudioDescriptor(backend=openai_backend).fit().predict([speech])[0].lower()
    assert any(word in description for word in SPEECH_WORDS)


def test_video_classifier_and_descriptor_see_the_apple(openai_backend, apple_video):
    """Classify and describe a video of an apple from its frames."""
    classifier = VideoClassifier(labels=['fruit', 'vehicle'], n_frames=3, backend=openai_backend).fit()
    assert classifier.predict([apple_video]).tolist() == ['fruit']
    description = VideoDescriptor(n_frames=3, backend=openai_backend).fit().predict([apple_video])[0].lower()
    assert 'apple' in description


def test_video_transcriber_reads_the_audio_track(openai_backend, apple_video):
    """Transcribe the speech of the video's audio track."""
    transcript = VideoTranscriber(language='en', backend=openai_backend).fit().predict([apple_video])[0].lower()
    assert 'apple' in transcript
    assert 'harvest' in transcript


def test_text_extractor_fills_fields(openai_backend):
    """Extract every field and leave the missing one empty."""
    fields = {'name': 'full name of the person', 'city': 'city of birth', 'year': 'year of birth', 'pet': 'pet name'}
    extractor = TextExtractor(fields=fields, backend=openai_backend).fit()
    record = extractor.predict(['Ada Lovelace was born in London in 1815.'])[0]
    assert 'Lovelace' in str(record['name'])
    assert 'London' in str(record['city'])
    assert '1815' in str(record['year'])
    assert record['pet'] is None


def test_image_editor_changes_the_image(openai_backend, shared_images):
    """Make the apple green and keep the image size."""
    apple = shared_images[0]
    edited = ImageEditor(instruction='Make the apple bright green.', backend=openai_backend).fit().predict([apple])[0]
    assert edited.size == apple.size
    description = ImageDescriptor(backend=openai_backend).fit().predict([edited])[0].lower()
    assert 'green' in description


def test_video_generation_and_animation(openai_backend, shared_images, video_enabled):
    """Generate a short video from a prompt and animate the apple image."""
    if not video_enabled:
        pytest.skip('Set SKAI_E2E_VIDEO=1 to run the video generation tests.')
    video = VideoGenerator(duration=4, size=(1280, 720), backend=openai_backend).fit().predict(['A paper boat on a pond'])[0]
    assert video.duration == pytest.approx(4, abs=0.2)
    assert video.frames.shape[1:3] == (720, 1280)
    animation = ImageAnimator(duration=4, backend=openai_backend).fit().predict([shared_images[0]])[0]
    assert animation.frames.shape[1:3] == (1024, 1024)
