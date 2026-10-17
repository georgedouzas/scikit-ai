"""Test the prompted tasks for audio, video and field extraction."""

import numpy as np
import pytest
from PIL import Image

from skai.core import (
    Audio,
    ClassificationRequest,
    DescriptionRequest,
    ExtractionRequest,
    InvalidResponseError,
    PromptedBackend,
    Video,
    sample_frames,
)


class ScriptedBackend(PromptedBackend):
    """Answer every prompt with a fixed answer and keep the prompts."""

    default_models = dict.fromkeys(
        ['classify_audio', 'classify_video', 'describe_audio', 'describe_video', 'extract_fields'],
        'scripted',
    )
    accepts_audio = True
    sends_video_audio = True

    def __init__(self, model=None, *, answer='', options=None, max_concurrency=1):
        """Store the fixed answer."""
        super().__init__(model, options=options, max_concurrency=max_concurrency)
        self.answer = answer

    def complete(self, prompt, model):
        """Keep the prompt and return the fixed answer."""
        self.prompts = [*getattr(self, 'prompts', []), prompt]
        return self.answer


class SilentBackend(ScriptedBackend):
    """Accept neither audio prompts nor the audio of videos."""

    accepts_audio = False
    sends_video_audio = False


@pytest.fixture
def frames(audio):
    """Return four frames sampled from a one-second video with audio."""
    return sample_frames(Video(np.zeros((25, 8, 8, 3), dtype=np.uint8), fps=25, audio=audio), 4)


def test_audio_classification_sends_audio(audio):
    """Send the audio of the sample and the examples and parse the label."""
    backend = ScriptedBackend(answer='siren')
    request = ClassificationRequest(audio, ('dog', 'siren'), examples=((audio, ('dog',)),))
    assert backend.classify_audio(request) == {'dog': 0.0, 'siren': 1.0}
    prompt = backend.prompts[-1]
    assert len(prompt.audio) == 2
    assert 'Input: <audio 1>\nOutput: dog' in prompt.user
    assert prompt.user.endswith('Input: <audio 2>\nOutput:')


def test_video_classification_sends_frames_and_audio(frames):
    """Send every sampled frame and the audio track."""
    backend = ScriptedBackend(answer='cooking')
    backend.classify_video(ClassificationRequest(frames, ('cooking', 'sports')))
    prompt = backend.prompts[-1]
    assert len(prompt.images) == 4
    assert len(prompt.audio) == 1
    assert 'video frames <image 1> <image 2> <image 3> <image 4> with audio <audio 1>' in prompt.user


def test_video_prompts_skip_audio_when_backend_does_not_send_it(frames):
    """Describe a video from its frames alone when the backend does not send video audio."""
    backend = SilentBackend(answer='A kitchen.')
    assert backend.describe_video(DescriptionRequest(frames, 'detailed')) == 'A kitchen.'
    prompt = backend.prompts[-1]
    assert prompt.audio == ()
    assert 'Describe the video in one detailed paragraph.' in prompt.user


def test_audio_description_names_audio(audio):
    """Ask for a description of the audio."""
    backend = ScriptedBackend(answer=' Birds singing. ')
    assert backend.describe_audio(DescriptionRequest(audio)) == 'Birds singing.'
    assert backend.prompts[-1].user.startswith('<audio 1>\n\nDescribe the audio in one sentence.')


def test_audio_tasks_require_audio_support():
    """Report prompted audio tasks only for backends that accept audio."""
    assert {'classify_audio', 'describe_audio'} <= ScriptedBackend.supported_tasks()
    assert not {'classify_audio', 'describe_audio'} & SilentBackend.supported_tasks()
    assert {'classify_video', 'describe_video', 'extract_fields'} <= SilentBackend.supported_tasks()


@pytest.mark.parametrize(
    ('answer', 'expected'),
    [
        ('{"name": "Ada", "city": "London"}', {'name': 'Ada', 'city': 'London'}),
        ('```json\n{"name": "Ada", "age": 36}\n```', {'name': 'Ada', 'city': None}),
    ],
)
def test_extraction_reads_json_answers(answer, expected):
    """Keep the requested fields, fill missing ones with `None` and drop unknown keys."""
    backend = ScriptedBackend(answer=answer)
    request = ExtractionRequest('Ada lives in London.', (('name', 'person name'), ('city', 'city of residence')))
    assert backend.extract_fields(request) == expected
    prompt = backend.prompts[-1]
    assert prompt.json_output
    assert '- city: city of residence' in prompt.user


@pytest.mark.parametrize('answer', ['not json', '["a list"]'])
def test_extraction_rejects_non_objects(answer):
    """Reject answers that are not JSON objects."""
    with pytest.raises(InvalidResponseError):
        ScriptedBackend(answer=answer).extract_fields(ExtractionRequest('text', (('name', 'person name'),)))


def test_image_examples_and_audio_share_numbering(image, audio):
    """Number images and audio separately in mixed prompts."""
    backend = ScriptedBackend(answer='cat')
    frames = sample_frames(Video(np.zeros((2, 4, 4, 3), dtype=np.uint8), fps=2, audio=audio), 2)
    backend.classify_video(ClassificationRequest(frames, ('cat',), examples=((frames, ('cat',)),)))
    prompt = backend.prompts[-1]
    assert len(prompt.images) == 4
    assert len(prompt.audio) == 2
    assert isinstance(prompt.images[0], Image.Image)
    assert isinstance(prompt.audio[0], Audio)
