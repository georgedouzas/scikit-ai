"""Test the prompts of prompted backends and the parsers of their answers."""

import json

import numpy as np
import pytest
from PIL import Image

from skai.core import (
    Audio,
    ClassificationRequest,
    ExtractionRequest,
    ImageTranscriptionRequest,
    InvalidResponseError,
    Moment,
    MomentRequest,
    Region,
    RegionRequest,
    TextGenerationRequest,
    ThumbnailRequest,
    UnsupportedBackendError,
    Video,
    build_task_prompt,
    sample_frames,
)
from skai.core._prompts import (
    build_detection_prompt,
    build_extraction_prompt,
    build_image_transcription_prompt,
    build_moment_prompt,
    build_text_generation_prompt,
    build_thumbnail_prompt,
    parse_fields,
    parse_frame_number,
    parse_labels,
    parse_moments,
    parse_regions,
)
from tests.fakes import ScriptedBackend, SilentBackend

MALFORMED = ['not json', '{"regions": [', '```json\n{oops}\n```']


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


def test_multimodal_classification_names_every_part(image):
    """Name every part of a multimodal sample by its key, with images referred to by number."""
    backend = ScriptedBackend(answer='fake')
    sample = {'title': 'Free money', 'thumbnail': image}
    assert backend.classify_multimodal(ClassificationRequest(sample, ('fake', 'real'))) == {'fake': 1.0, 'real': 0.0}
    prompt = backend.prompts[-1]
    assert 'Input: title: Free money\nthumbnail: <image 1>' in prompt.user
    assert prompt.images == (image,)


def test_multi_label_answers_name_several_labels():
    """Score every named label, and none for the no-label answer."""
    assert parse_labels('a, c', ('a', 'b', 'c'), multi_label=True) == {'a': 1.0, 'b': 0.0, 'c': 1.0}
    assert parse_labels('none', ('a', 'b'), multi_label=True) == {'a': 0.0, 'b': 0.0}
    with pytest.raises(InvalidResponseError, match='exactly one of the labels'):
        parse_labels('a and b', ('a', 'b'), multi_label=False)


def test_video_prompts_skip_audio_when_backend_does_not_send_it(frames):
    """Generate text from the frames alone when the backend does not send video audio."""
    backend = SilentBackend(answer='A kitchen.')
    backend.default_models = {'generate_text': 'silent'}
    assert backend.generate_text(TextGenerationRequest(frames, 'Describe the video in detail.')) == 'A kitchen.'
    prompt = backend.prompts[-1]
    assert prompt.audio == ()
    assert 'Instructions: Describe the video in detail.' in prompt.user
    assert 'with audio' not in prompt.user


def test_text_generation_from_audio_strips_the_answer(audio):
    """Send the audio and return the stripped answer."""
    backend = ScriptedBackend(answer=' Birds singing. ')
    assert backend.generate_text(TextGenerationRequest(audio, 'Describe the audio.')) == 'Birds singing.'
    prompt = backend.prompts[-1]
    assert prompt.user.startswith('Instructions: Describe the audio.')
    assert prompt.user.endswith('Input: <audio 1>\nOutput:')
    assert len(prompt.audio) == 1


def test_text_generation_prompt_states_length_glossary_and_examples():
    """Render the instructions, the length limit, the glossary and the examples in order."""
    request = TextGenerationRequest(
        'Open SkaiBox.',
        'Translate into French.',
        examples=(('Hi.', 'Salut.'),),
        max_length=5,
        glossary=(('SkaiBox', 'SkaiBox'),),
    )
    prompt = build_text_generation_prompt(request)
    assert prompt.user == (
        'Instructions: Translate into French.\nUse at most 5 words.\nRender these terms as given:\n'
        '- SkaiBox → SkaiBox\n\nExample 1:\nInput: Hi.\nOutput: Salut.\n\nInput: Open SkaiBox.\nOutput:'
    )
    assert not prompt.json_output


def test_prompted_backend_refuses_audio_without_audio_support(audio, image):
    """Refuse audio samples, alone or inside a multimodal sample, unless the backend accepts audio."""
    backend = SilentBackend(answer='ok')
    with pytest.raises(UnsupportedBackendError, match='does not accept audio for `generate_text`'):
        backend.generate_text(TextGenerationRequest(audio, 'Describe the audio.'))
    with pytest.raises(UnsupportedBackendError, match='does not accept audio for `extract_fields`'):
        backend.extract_fields(ExtractionRequest({'voice': audio, 'cover': image}, (('name', 'str'),)))
    assert not hasattr(backend, 'prompts')


def test_prompted_backend_supports_audio_only_when_it_accepts_audio():
    """Report prompted tasks on audio samples only for backends that accept audio."""
    assert ScriptedBackend.supports('generate_text', 'audio')
    assert ScriptedBackend.supports('classify_audio')
    assert not SilentBackend.supports('generate_text', 'audio')
    assert not SilentBackend.supports('find_moments', 'audio')
    assert SilentBackend.supports('generate_text', 'image')
    assert SilentBackend.supports('find_moments', 'video')
    assert SilentBackend.supports('generate_text')
    assert 'classify_audio' not in SilentBackend.supported_tasks()
    assert not SilentBackend.supports('transcribe')


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
    assert 'Template:\n{"name": "<person name>", "city": "<city of residence>"}' in prompt.user
    assert prompt.user.endswith('Input:\nAda lives in London.')


def test_extraction_prompt_renders_nested_and_list_fields():
    """Turn nested dicts and lists of declarations into a JSON template."""
    fields = (('vendor', {'name': 'company name', 'country': 'country'}), ('items', [{'sku': 'code', 'qty': 'int'}]))
    prompt = build_extraction_prompt(ExtractionRequest('An invoice.', fields))
    template = prompt.user.split('Template:\n')[1].split('\n')[0]
    assert json.loads(template) == {
        'vendor': {'name': '<company name>', 'country': '<country>'},
        'items': [{'sku': '<code>', 'qty': '<int>'}],
    }


def test_extraction_prompt_states_instructions_and_examples(image):
    """Start with the instructions and render every example with its JSON answer."""
    request = ExtractionRequest(
        image,
        (('total', 'number'),),
        instructions='Read amounts in euros.',
        examples=((image, {'total': 12}),),
    )
    prompt = build_extraction_prompt(request)
    assert prompt.user.startswith('Instructions: Read amounts in euros.\n\nTemplate:')
    assert 'Example input:\n<image 1>\nExample answer:\n{"total": 12}' in prompt.user
    assert prompt.user.endswith('Input:\n<image 2>')
    assert len(prompt.images) == 2


@pytest.mark.parametrize('answer', ['not json', '["a list"]'])
def test_extraction_rejects_non_objects(answer):
    """Reject answers that are not JSON objects."""
    with pytest.raises(InvalidResponseError):
        ScriptedBackend(answer=answer).extract_fields(ExtractionRequest('text', (('name', 'person name'),)))


def test_parse_fields_keeps_nested_values():
    """Keep nested values as given and drop unknown keys."""
    answer = '{"vendor": {"name": "Acme"}, "items": [{"sku": "A1"}], "extra": 1}'
    assert parse_fields(answer, ('vendor', 'items', 'total')) == {
        'vendor': {'name': 'Acme'},
        'items': [{'sku': 'A1'}],
        'total': None,
    }


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


def test_image_transcription_prompt_asks_for_markdown_only_when_requested(image):
    """Ask for the exact text, keeping the page structure as Markdown when requested."""
    plain = build_image_transcription_prompt(ImageTranscriptionRequest(image))
    markdown = build_image_transcription_prompt(ImageTranscriptionRequest(image, markdown=True))
    assert plain.images == (image,)
    assert plain.user.startswith('<image 1>\n\nTranscribe all the text exactly as written.')
    assert 'Markdown' not in plain.user
    assert 'Markdown headings, lists and tables' in markdown.user


def test_image_transcription_returns_stripped_text(image):
    """Return the stripped answer of the vision model."""
    assert ScriptedBackend(answer='\n# Title\n').transcribe_image(ImageTranscriptionRequest(image)) == '# Title'


def test_detection_prompt_states_size_and_targets(image):
    """State the image size and the labels, or the prompt when there are no labels."""
    by_labels = build_detection_prompt(RegionRequest(image, labels=('cat', 'dog')))
    by_prompt = build_detection_prompt(RegionRequest(image, prompt='the speaker'))
    assert by_labels.json_output
    assert 'The image is 32x24 pixels. Find every region showing: cat, dog.' in by_labels.user
    assert 'Find every region showing: the speaker.' in by_prompt.user
    assert by_labels.images == (image,)


def test_parse_regions_clamps_boxes_to_the_image():
    """Read labels, scores and boxes, clamping the boxes to the image."""
    regions = '[{"label": "cat", "box": [-5, 2.4, 50, 30], "score": 0.8}, {"box": [1, 1, 4, 4]}]'
    answer = f'```json\n{{"regions": {regions}}}\n```'
    regions = parse_regions(answer, (32, 24))
    assert regions == [Region('cat', 0.8, (0, 2, 32, 24)), Region('', 1.0, (1, 1, 4, 4))]


@pytest.mark.parametrize('answer', MALFORMED)
def test_parse_regions_rejects_malformed_json(answer):
    """Report an answer that is not valid JSON."""
    with pytest.raises(InvalidResponseError, match='not valid JSON'):
        parse_regions(answer, (8, 8))


def test_parse_regions_requires_a_list_of_regions():
    """Report a JSON answer without a list of regions."""
    with pytest.raises(InvalidResponseError, match='no list of regions'):
        parse_regions('{"boxes": []}', (8, 8))


def test_detection_returns_regions(image):
    """Parse the regions of the answer of the vision model."""
    backend = ScriptedBackend(answer='{"regions": [{"label": "cat", "box": [0, 0, 10, 10], "score": 0.5}]}')
    assert backend.detect_objects(RegionRequest(image, labels=('cat',))) == [Region('cat', 0.5, (0, 0, 10, 10))]


def test_moment_prompt_states_frame_times(frames):
    """State the length of the video and the time of every frame."""
    prompt = build_moment_prompt(MomentRequest(frames, 'a goal'))
    assert prompt.json_output
    assert 'The video lasts 1.0s, with frame 1 at 0.0s, frame 2 at 0.2s' in prompt.user
    assert 'Find every moment that matches: a goal.' in prompt.user
    assert len(prompt.images) == 4
    assert len(prompt.audio) == 1
    assert build_moment_prompt(MomentRequest(frames, 'a goal'), include_audio=False).audio == ()


def test_moment_prompt_states_audio_length(audio):
    """State the length of audio and send it."""
    prompt = build_moment_prompt(MomentRequest(audio, 'laughter'))
    assert prompt.user.startswith('<audio 1>\n\nThe audio lasts 1.0s.')
    assert prompt.audio == (audio,)


def test_parse_moments_clamps_sorts_and_drops_empty_moments():
    """Clamp moments to the media, sort them by start and drop moments that end before they start."""
    answer = json.dumps(
        {'moments': [{'start': 5, 'end': 20, 'score': 0.7}, {'start': -1, 'end': 2}, {'start': 4, 'end': 3}]},
    )
    assert parse_moments(answer, 10.0) == [Moment(0.0, 2.0, 1.0), Moment(5.0, 10.0, 0.7)]


@pytest.mark.parametrize('answer', MALFORMED)
def test_parse_moments_rejects_malformed_json(answer):
    """Report an answer that is not valid JSON."""
    with pytest.raises(InvalidResponseError, match='not valid JSON'):
        parse_moments(answer, 1.0)


def test_parse_moments_requires_a_list_of_moments():
    """Report a JSON answer without a list of moments."""
    with pytest.raises(InvalidResponseError, match='no list of moments'):
        parse_moments('{"moments": {}}', 1.0)


def test_find_moments_uses_the_length_of_the_sample(audio):
    """Clamp the moments of the answer to the audio."""
    backend = ScriptedBackend(answer='{"moments": [{"start": 0.5, "end": 3}]}')
    assert backend.find_moments(MomentRequest(audio, 'a beep')) == [Moment(0.5, 1.0, 1.0)]


def test_thumbnail_prompt_numbers_every_frame(frames):
    """Refer to every frame by number and ask for one of them."""
    prompt = build_thumbnail_prompt(ThumbnailRequest(frames))
    assert prompt.user.startswith('<image 1> <image 2> <image 3> <image 4>\n\nWhich frame, from 1 to 4,')
    assert prompt.images == frames.frames


@pytest.mark.parametrize(('answer', 'index'), [('3', 2), ('Frame 1.', 0), ('4', 3)])
def test_parse_frame_number_returns_zero_based_index(answer, index):
    """Read the first number of the answer as a zero-based index."""
    assert parse_frame_number(answer, 4) == index


@pytest.mark.parametrize('answer', ['0', '5', 'the first one'])
def test_parse_frame_number_rejects_missing_or_out_of_range_frames(answer):
    """Report an answer that names no frame in range."""
    with pytest.raises(InvalidResponseError, match='names no frame between 1 and 4'):
        parse_frame_number(answer, 4)


def test_pick_thumbnail_returns_the_named_frame(frames):
    """Return the index of the frame the vision model names."""
    assert ScriptedBackend(answer='2').pick_thumbnail(ThumbnailRequest(frames)) == 1


def test_build_task_prompt_dispatches_by_task(image):
    """Build the prompt of every prompted task with a fine-tunable request, and reject other tasks."""
    classify = build_task_prompt('classify_multimodal', ClassificationRequest({'a': 'text'}, ('x', 'y')))
    generate = build_task_prompt('generate_text', TextGenerationRequest(image, 'Describe the image.'))
    extract = build_task_prompt('extract_fields', ExtractionRequest('text', (('name', 'str'),)))
    assert classify.user.endswith('Input: a: text\nOutput:')
    assert generate.images == (image,)
    assert extract.json_output
    with pytest.raises(ValueError, match='`transcribe` is not answered by prompting'):
        build_task_prompt('transcribe', TextGenerationRequest('text', 'x'))
