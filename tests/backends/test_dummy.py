"""Test the offline dummy backend."""

import numpy as np
import pytest
from PIL import Image

from skai.backends import DummyBackend
from skai.classification import ImageClassifier, MultimodalClassifier, TextClassifier
from skai.core import (
    TASKS,
    AnimationRequest,
    Audio,
    AudioGenerationRequest,
    AudioOperationRequest,
    AvatarRequest,
    ClassificationRequest,
    DiarizationRequest,
    EmbeddingRequest,
    ExtractionRequest,
    FineTuneRequest,
    ImageEditRequest,
    ImageGenerationRequest,
    ImageOperationRequest,
    ImageTranscriptionRequest,
    LipSyncRequest,
    Moment,
    MomentRequest,
    Prompt,
    Region,
    RegionRequest,
    Segment,
    SpeechRequest,
    TextGenerationRequest,
    ThumbnailRequest,
    Transcript,
    TranscriptionRequest,
    Video,
    VideoGenerationRequest,
    sample_frames,
)
from skai.embedding import TextEmbedder
from skai.generation import (
    AudioToTextGenerator,
    AudioToVideoGenerator,
    ImageToTextGenerator,
    TextToAudioGenerator,
    TextToImageGenerator,
    TextToTextGenerator,
)

IMAGE = Image.new('RGB', (32, 24), (200, 30, 30))
AUDIO = Audio((0.1 * np.sin(np.linspace(0, 880 * np.pi, 16000))).astype(np.float32), 16000)
VIDEO = Video(np.zeros((10, 8, 8, 3), dtype=np.uint8), fps=5)
FRAMES = sample_frames(VIDEO, 4)
COMPONENTS = [
    (TextClassifier(labels=['a', 'b'], backend='dummy'), ['x', 'y']),
    (MultimodalClassifier(labels=['a', 'b'], backend='dummy'), [{'title': 'x', 'cover': IMAGE}]),
    (TextToTextGenerator(instructions='Summarize the text.', backend='dummy'), ['x y z']),
    (TextToTextGenerator(instructions='Translate into German.', learn_terms=True, backend='dummy'), ['hello']),
    (TextEmbedder(backend='dummy'), ['hello']),
    (TextToAudioGenerator(backend='dummy', random_state=0), ['hello']),
    (TextToImageGenerator(size=(8, 8), backend='dummy', random_state=0), ['a cat']),
    (ImageToTextGenerator(instructions='Describe the image.', backend='dummy'), [Image.new('RGB', (8, 8))]),
    (ImageClassifier(labels=['cat', 'dog'], backend='dummy'), [Image.new('RGB', (8, 8))]),
    (AudioToTextGenerator(backend='dummy'), [(np.zeros(1600), 16000)]),
    (AudioToVideoGenerator(avatar=Image.new('RGB', (8, 8)), backend='dummy'), [(np.zeros(1600), 16000)]),
]
LABELS = ('a', 'b', 'c')
REQUESTS = {
    'classify_text': ClassificationRequest('a text', LABELS),
    'classify_image': ClassificationRequest(IMAGE, LABELS),
    'classify_audio': ClassificationRequest(AUDIO, LABELS),
    'classify_video': ClassificationRequest(FRAMES, LABELS),
    'classify_multimodal': ClassificationRequest({'title': 'a text', 'cover': IMAGE}, LABELS, multi_label=True),
    'generate_text': TextGenerationRequest(IMAGE, 'Describe the image.'),
    'extract_fields': ExtractionRequest('Ada lives in London.', (('name', 'str'), ('tags', ['str']))),
    'transcribe': TranscriptionRequest(AUDIO, timestamps='segment'),
    'transcribe_image': ImageTranscriptionRequest(IMAGE),
    'embed_text': EmbeddingRequest('a text'),
    'embed_image': EmbeddingRequest(IMAGE, 16),
    'embed_audio': EmbeddingRequest(AUDIO),
    'detect_objects': RegionRequest(IMAGE, labels=('cat', 'dog')),
    'segment_image': RegionRequest(IMAGE, prompt='the cat'),
    'diarize': DiarizationRequest(AUDIO),
    'find_moments': MomentRequest(FRAMES, 'a goal'),
    'synthesize_speech': SpeechRequest('hello there', seed=1),
    'generate_music': AudioGenerationRequest('calm piano', 1.5, seed=1),
    'generate_sound': AudioGenerationRequest('rain', 0.5, seed=1),
    'separate_audio': AudioOperationRequest(AUDIO),
    'enhance_audio': AudioOperationRequest(AUDIO),
    'convert_voice': AudioOperationRequest(AUDIO, voice=AUDIO),
    'generate_image': ImageGenerationRequest('a cat', (16, 8), seed=1),
    'edit_image': ImageEditRequest(IMAGE, 'add snow', seed=1),
    'upscale_image': ImageOperationRequest(IMAGE, scale=3),
    'remove_background': ImageOperationRequest(IMAGE),
    'pick_thumbnail': ThumbnailRequest(FRAMES),
    'generate_video': VideoGenerationRequest('a boat', duration=2, size=(32, 24), fps=8),
    'animate_image': AnimationRequest(IMAGE, duration=1.5, fps=4),
    'animate_avatar': AvatarRequest(IMAGE, AUDIO),
    'lip_sync': LipSyncRequest(VIDEO, AUDIO),
    'fine_tune': FineTuneRequest('classify_text', ((Prompt('system', 'user'), 'a'),)),
}


def check_classification(scores, request):
    """Score every label in `[0, 1]`, summing to one for a single label."""
    assert set(scores) == set(request.labels)
    assert all(0 <= score <= 1 for score in scores.values())
    if not request.multi_label:
        assert sum(scores.values()) == pytest.approx(1)


def check_embedding(vector, request):
    """Return a vector of the requested or the default size."""
    assert isinstance(vector, np.ndarray)
    assert vector.shape == (request.dimensions or 8,)


def check_regions(regions, request):
    """Return regions inside the image, with a mask of the image size for segmentation."""
    assert regions
    width, height = request.image.size
    for region in regions:
        assert isinstance(region, Region)
        left, top, right, bottom = region.box
        assert 0 <= left < right <= width
        assert 0 <= top < bottom <= height
        assert region.mask is None or region.mask.size == request.image.size


CHECKS = {
    'classify_text': check_classification,
    'classify_image': check_classification,
    'classify_audio': check_classification,
    'classify_video': check_classification,
    'classify_multimodal': check_classification,
    'generate_text': lambda text, _: text == 'A 32x24 image.',
    'extract_fields': lambda record, _: set(record) == {'name', 'tags'}
    and isinstance(record['name'], str)
    and len(record['tags']) == 2,
    'transcribe': lambda transcript, _: isinstance(transcript, Transcript) and len(transcript.segments) == 1,
    'transcribe_image': lambda text, _: text == 'Text of a 32x24 image.',
    'embed_text': check_embedding,
    'embed_image': check_embedding,
    'embed_audio': check_embedding,
    'detect_objects': lambda regions, request: check_regions(regions, request) is None
    and [region.label for region in regions] == ['cat', 'dog'],
    'segment_image': lambda regions, request: check_regions(regions, request) is None
    and all(region.mask is not None for region in regions),
    'diarize': lambda transcript, _: [segment.speaker for segment in transcript.segments] == ['SPEAKER_0', 'SPEAKER_1']
    and transcript.segments[-1].end == pytest.approx(1.0),
    'find_moments': lambda moments, _: moments == [Moment(2 / 3, 4 / 3, 0.9)],
    'synthesize_speech': lambda audio, _: isinstance(audio, Audio) and audio.duration == pytest.approx(0.4),
    'generate_music': lambda audio, request: audio.duration == pytest.approx(request.duration),
    'generate_sound': lambda audio, request: audio.duration == pytest.approx(request.duration),
    'separate_audio': lambda audio, request: audio.duration == request.audio.duration,
    'enhance_audio': lambda audio, request: audio.duration == request.audio.duration,
    'convert_voice': lambda audio, request: audio.duration == pytest.approx(request.audio.duration),
    'generate_image': lambda image, request: isinstance(image, Image.Image) and image.size == request.size,
    'edit_image': lambda image, request: image.size == request.image.size,
    'upscale_image': lambda image, _: image.size == (96, 72),
    'remove_background': lambda image, request: image.mode == 'RGBA' and image.size == request.image.size,
    'pick_thumbnail': lambda index, request: isinstance(index, int) and 0 <= index < len(request.frames.frames),
    'generate_video': lambda video, _: (video.duration, video.frames.shape[1:3]) == (2.0, (24, 32)),
    'animate_image': lambda video, _: video.frames.shape[0] == 6,
    'animate_avatar': lambda video, _: video.duration == pytest.approx(1.0) and video.audio is not None,
    'lip_sync': lambda video, request: video.duration == pytest.approx(request.audio.duration),
    'fine_tune': lambda model, _: model.startswith('dummy-finetuned-'),
}


def comparable(output):  # noqa: PLR0911 - one return per output type
    """Return an output as a value that compares by content."""
    if isinstance(output, Image.Image):
        return (output.mode, output.size, output.tobytes())
    if isinstance(output, Audio):
        return (output.sample_rate, output.samples.tobytes())
    if isinstance(output, Video):
        return (output.fps, output.frames.tobytes())
    if isinstance(output, np.ndarray):
        return output.tobytes()
    if isinstance(output, list):
        return [comparable(item) for item in output]
    if isinstance(output, Region):
        return (output.label, output.score, output.box, comparable(output.mask) if output.mask else None)
    return output


def outputs_of(component, X):
    """Return the outputs of a fitted component as comparable values."""
    output = component.transform(X) if isinstance(component, TextEmbedder) else component.predict(X)
    return [np.asarray(item.frames if hasattr(item, 'frames') else getattr(item, 'samples', item)) for item in output]


def test_supported_tasks_cover_every_task():
    """Support every task on every input kind."""
    assert DummyBackend.supported_tasks() == frozenset(TASKS)
    assert set(REQUESTS) == set(TASKS)
    assert all(DummyBackend.supports(task, 'audio') for task in TASKS)


@pytest.mark.parametrize('task', TASKS)
def test_every_task_returns_deterministic_typed_outputs(task):
    """Return an output of the documented type, the same one for the same request."""
    backend = DummyBackend()
    request = REQUESTS[task]
    output = getattr(backend, task)(request)
    assert CHECKS[task](output, request) in {None, True}
    assert comparable(getattr(DummyBackend(), task)(request)) == comparable(output)


def test_outputs_depend_on_the_request():
    """Return different outputs for different prompts, seeds and samples."""
    backend = DummyBackend()
    cat = backend.generate_image(ImageGenerationRequest('a cat', (4, 4), seed=1))
    dog = backend.generate_image(ImageGenerationRequest('a dog', (4, 4), seed=1))
    assert cat.tobytes() != dog.tobytes()
    first = backend.embed_text(EmbeddingRequest('one'))
    assert not np.array_equal(first, backend.embed_text(EmbeddingRequest('two')))


def test_text_generation_follows_length_and_glossary():
    """Return the text with the glossary applied and cut to the maximum length."""
    backend = DummyBackend()
    request = TextGenerationRequest('Open the box now please.', 'x', max_length=3, glossary=(('box', 'SkaiBox'),))
    assert backend.generate_text(request) == 'Open the SkaiBox'
    assert backend.generate_text(TextGenerationRequest(AUDIO, 'x')) == '1.00 seconds of audio.'
    assert backend.generate_text(TextGenerationRequest(FRAMES, 'x')) == '4 frames of a 2.00 second video.'
    multimodal = backend.generate_text(TextGenerationRequest({'title': 'Hi', 'cover': IMAGE}, 'x'))
    assert multimodal == 'title: Hi cover: A 32x24 image.'


def test_extraction_follows_nested_declarations():
    """Fill nested fields with words of the sample and lists with two items."""
    request = ExtractionRequest('Ada lives in London.', (('person', {'name': 'str', 'city': 'str'}), ('tags', ['str'])))
    record = DummyBackend().extract_fields(request)
    assert set(record['person']) == {'name', 'city'}
    assert all(value in request.sample.split() for value in record['person'].values())
    assert len(record['tags']) == 2


def test_transcription_timestamps_and_vocabulary():
    """Return word segments that cover the audio and mention the vocabulary."""
    transcript = DummyBackend().transcribe(TranscriptionRequest(AUDIO, timestamps='word', vocabulary=('gRPC',)))
    assert 'gRPC' in transcript.text
    assert len(transcript.segments) == len(transcript.text.split())
    assert transcript.segments[-1].end == pytest.approx(1.0)
    assert all(isinstance(segment, Segment) for segment in transcript.segments)
    assert DummyBackend().transcribe(TranscriptionRequest(AUDIO)).segments == ()


def test_image_transcription_returns_markdown_when_asked():
    """Return a Markdown heading when the request asks for Markdown."""
    assert DummyBackend().transcribe_image(ImageTranscriptionRequest(IMAGE, markdown=True)).startswith('# ')


def test_mask_limits_image_edits():
    """Keep the image outside the mask unchanged."""
    mask = Image.new('L', IMAGE.size)
    backend = DummyBackend()
    assert backend.edit_image(ImageEditRequest(IMAGE, 'add snow', mask=mask)).tobytes() == IMAGE.tobytes()
    assert backend.edit_image(ImageEditRequest(IMAGE, 'add snow')).tobytes() != IMAGE.tobytes()


def test_audio_operations_keep_the_length():
    """Return a stem at half the volume, as long as the input."""
    stem = DummyBackend().separate_audio(AudioOperationRequest(AUDIO))
    np.testing.assert_allclose(stem.samples, AUDIO.samples * 0.5)


@pytest.mark.parametrize(('component', 'X'), COMPONENTS)
def test_outputs_are_deterministic(component, X):
    """Return identical outputs for identical requests without network access."""
    first = outputs_of(component.fit(X if isinstance(component, TextEmbedder) else None), X)
    second = outputs_of(component, X)
    assert len(first) == len(X)
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(np.asarray(left, dtype=object), np.asarray(right, dtype=object))


def test_avatar_duration_matches_audio():
    """Return a video as long as its audio."""
    generator = AudioToVideoGenerator(avatar=Image.new('RGB', (128, 96)), backend='dummy').fit()
    video = generator.predict([(np.zeros(24000), 16000)])[0]
    assert video.duration == pytest.approx(1.5, abs=1 / 25)
    assert max(video.frames.shape[1:3]) <= 64
