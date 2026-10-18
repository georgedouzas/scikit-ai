"""Run the audio, video, extraction and editing components on the OpenAI backend."""

import pytest

from skai.classification import AudioClassifier, VideoClassifier
from skai.extraction import TextExtractor
from skai.generation import (
    AudioToTextGenerator,
    ImageToImageGenerator,
    ImageToTextGenerator,
    ImageToVideoGenerator,
    TextToVideoGenerator,
    VideoToTextGenerator,
)

SPEECH_WORDS = ('speech', 'speak', 'voice', 'talk', 'say', 'announce', 'greet', 'apple', 'harvest')


def test_audio_classifier_recognizes_speech(openai_backend, shared_speech):
    """Label synthesized speech as speech."""
    classifier = AudioClassifier(labels=['speech', 'music', 'birdsong'], backend=openai_backend).fit()
    assert classifier.predict([shared_speech]).tolist() == ['speech']


def test_audio_descriptor_mentions_speech(openai_backend, shared_speech):
    """Describe synthesized speech by what is said or by the act of speaking."""
    descriptor = AudioToTextGenerator(
        operation='instruct',
        instructions='Describe the audio in one sentence.',
        backend=openai_backend,
    )
    description = descriptor.fit().predict([shared_speech])[0].lower()
    assert any(word in description for word in SPEECH_WORDS)


def test_video_classifier_and_descriptor_see_the_apple(openai_backend, apple_video):
    """Classify and describe a video of an apple from its frames."""
    classifier = VideoClassifier(labels=['fruit', 'vehicle'], n_frames=3, backend=openai_backend).fit()
    assert classifier.predict([apple_video]).tolist() == ['fruit']
    instructions = 'Describe the video in one sentence.'
    descriptor = VideoToTextGenerator(instructions=instructions, n_frames=3, backend=openai_backend)
    assert 'apple' in descriptor.fit().predict([apple_video])[0].lower()


def test_video_transcriber_reads_the_audio_track(openai_backend, apple_video):
    """Transcribe the speech of the video's audio track."""
    transcriber = VideoToTextGenerator(operation='transcribe', language='en', backend=openai_backend)
    transcript = transcriber.fit().predict([apple_video])[0].lower()
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
    instructions = 'Make the apple bright green.'
    editor = ImageToImageGenerator(operation='edit', instructions=instructions, backend=openai_backend)
    edited = editor.fit().predict([apple])[0]
    assert edited.size == apple.size
    describer = ImageToTextGenerator(instructions='Describe the image in one sentence.', backend=openai_backend)
    assert 'green' in describer.fit().predict([edited])[0].lower()


def test_video_generation_and_animation(openai_backend, shared_images, video_enabled):
    """Generate a short video from a prompt and animate the apple image."""
    if not video_enabled:
        pytest.skip('Set SKAI_E2E_VIDEO=1 to run the video generation tests.')
    generator = TextToVideoGenerator(duration=4, size=(1280, 720), backend=openai_backend).fit()
    video = generator.predict(['A paper boat on a pond'])[0]
    assert video.duration == pytest.approx(4, abs=0.2)
    assert video.frames.shape[1:3] == (720, 1280)
    animation = ImageToVideoGenerator(duration=4, backend=openai_backend).fit().predict([shared_images[0]])[0]
    assert animation.frames.shape[1:3] == (1024, 1024)
