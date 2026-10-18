"""Run every operation of the component grid that the OpenAI backend supports, beyond the 0.2 tasks."""

import pytest
from PIL import Image

from skai import presets
from skai.classification import ImageClassifier, MultimodalClassifier
from skai.core import Audio, UnsupportedBackendError, Video
from skai.extraction import AudioExtractor, ImageExtractor, MultimodalExtractor, TextExtractor, VideoExtractor
from skai.generation import (
    AudioToTextGenerator,
    ImageToTextGenerator,
    MultimodalToAudioGenerator,
    MultimodalToImageGenerator,
    MultimodalToTextGenerator,
    TextToImageGenerator,
    VideoToImageGenerator,
    VideoToTextGenerator,
    VideoToVideoGenerator,
)
from skai.segmentation import AudioSegmenter, ImageSegmenter, VideoSegmenter

TOLERANCE = 0.5


def _names(items):
    """Return the lowercase text of every item of an extracted list."""
    return ' '.join(str(item).lower() for item in items or [])


def _check_timed(records, duration):
    """Check that timed records are ordered pairs inside the media."""
    for record in records:
        assert 0 <= record['start'] <= record['end'] <= duration + TOLERANCE


# Text generation from every input kind


def test_text_question_is_answered(openai_backend):
    """Answer a question about a text."""
    answerer = presets.question_answerer('Which city does Ada live in?', backend=openai_backend).fit()
    assert 'london' in answerer.predict(['Ada moved from Paris and now lives in London.'])[0].lower()


def test_image_question_is_answered(openai_backend, shared_images):
    """Answer a question about an image."""
    answerer = presets.question_answerer('Which fruit is shown?', kind='image', backend=openai_backend).fit()
    assert 'apple' in answerer.predict(shared_images[:1])[0].lower()


def test_audio_instructions_are_followed(openai_backend, shared_speech):
    """Answer a question about speech with the audio model."""
    generator = AudioToTextGenerator(
        operation='instruct',
        instructions='Which fruit does the speaker mention? Answer with one word.',
        backend=openai_backend,
    )
    assert 'apple' in generator.fit().predict([shared_speech])[0].lower()


def test_video_instructions_are_followed(openai_backend, scenes_video):
    """List the things that a video of two shots shows."""
    generator = VideoToTextGenerator(
        instructions='List the main objects that the video shows, in order.',
        n_frames=4,
        backend=openai_backend,
    )
    answer = generator.fit().predict([scenes_video])[0].lower()
    assert 'apple' in answer
    assert 'car' in answer


def test_multimodal_parts_are_combined(openai_backend, shared_images):
    """Answer the question of a sample about the image of the same sample."""
    generator = MultimodalToTextGenerator(
        instructions='Answer the question about the image in a few words.',
        backend=openai_backend,
    )
    samples = [{'image': shared_images[1], 'question': 'What vehicle is shown, and what color is it?'}]
    answer = generator.fit().predict(samples)[0].lower()
    assert 'car' in answer
    assert 'blue' in answer


# Extraction with nested list fields


def test_text_extractor_fills_nested_lists(openai_backend):
    """Extract one record per person of a text."""
    fields = {'people': [{'name': 'full name of the person', 'city': 'city where the person lives'}]}
    extractor = TextExtractor(fields=fields, backend=openai_backend).fit()
    record = extractor.predict(['Ada Lovelace lives in London, and Alan Turing lives in Manchester.'])[0]
    people = record['people']
    assert isinstance(people, list)
    assert len(people) == 2
    assert all(isinstance(person, dict) for person in people)
    assert 'lovelace' in _names(people)
    assert 'manchester' in _names(people)


def test_image_extractor_lists_objects(openai_backend, shared_images):
    """Extract the objects of an image with their colors."""
    fields = {'objects': [{'name': 'name of the object', 'color': 'main color of the object'}]}
    record = ImageExtractor(fields=fields, backend=openai_backend).fit().predict(shared_images[:1])[0]
    assert isinstance(record['objects'], list)
    assert 'apple' in _names(record['objects'])
    assert 'red' in _names(record['objects'])


def test_audio_extractor_reads_speech(openai_backend, shared_speech):
    """Extract the greeting and the events of speech."""
    fields = {'greeting': 'the greeting that is said', 'events': [{'event': 'what happens', 'when': 'when it happens'}]}
    record = AudioExtractor(fields=fields, backend=openai_backend).fit().predict([shared_speech])[0]
    assert 'morning' in str(record['greeting']).lower()
    assert isinstance(record['events'], list)
    assert 'harvest' in _names(record['events'])


def test_video_extractor_lists_objects(openai_backend, scenes_video):
    """Extract the objects shown in a video of two shots."""
    fields = {'objects': [{'name': 'name of the object'}]}
    record = VideoExtractor(fields=fields, n_frames=4, backend=openai_backend).fit().predict([scenes_video])[0]
    assert isinstance(record['objects'], list)
    assert 'apple' in _names(record['objects'])
    assert 'car' in _names(record['objects'])


def test_multimodal_extractor_combines_parts(openai_backend, shared_images):
    """Extract a listing from a photo and its caption."""
    fields = {'vehicle': 'kind of vehicle', 'color': 'color of the vehicle', 'price': 'asking price'}
    sample = {'image': shared_images[1], 'caption': 'For sale, 5000 euros, barely used.'}
    record = MultimodalExtractor(fields=fields, backend=openai_backend).fit().predict([sample])[0]
    assert 'car' in str(record['vehicle']).lower()
    assert 'blue' in str(record['color']).lower()
    assert '5' in str(record['price'])


# Classification of multimodal samples


def test_multimodal_classifier_judges_image_and_caption(openai_backend, shared_images):
    """Tell whether every caption matches its image."""
    classifier = MultimodalClassifier(
        labels=['matching', 'mismatching'],
        instructions='Decide whether the caption describes the image.',
        backend=openai_backend,
    ).fit()
    apple = shared_images[0]
    posts = [{'image': apple, 'caption': 'A red apple on a table.'}, {'image': apple, 'caption': 'A blue car.'}]
    assert classifier.predict(posts).tolist() == ['matching', 'mismatching']


# Image transcription and object detection


def test_image_transcription_reads_generated_text(openai_backend):
    """Transcribe the words of a generated poster as Markdown."""
    prompt = 'A plain white poster with the words "FRESH APPLES" in large black letters'
    poster = TextToImageGenerator(backend=openai_backend).fit().predict([prompt])[0]
    transcriber = ImageToTextGenerator(operation='transcribe', format='markdown', backend=openai_backend)
    text = transcriber.fit().predict([poster])[0].lower()
    assert 'fresh' in text
    assert 'apples' in text


def test_image_segmenter_finds_the_apple(openai_backend, shared_images):
    """Find the box of the apple and crop it."""
    apple = shared_images[0]
    regions = ImageSegmenter(by='object', labels=['apple'], backend=openai_backend).fit().predict([apple])[0]
    assert regions
    width, height = apple.size
    for region in regions:
        assert region['label'] == 'apple'
        left, top, right, bottom = region['box']
        assert 0 <= left < right <= width
        assert 0 <= top < bottom <= height
        assert isinstance(region['media'], Image.Image)


# Timed regions of audio and video


def test_audio_segmenter_times_speech_and_words(openai_backend, shared_speech):
    """Time the speech and the words of a sentence."""
    for by in ('speech', 'word'):
        regions = AudioSegmenter(by=by, language='en', backend=openai_backend).fit().predict([shared_speech])[0]
        assert regions
        _check_timed(regions, shared_speech.duration)
        assert 'harvest' in ' '.join(region['text'] for region in regions).lower()
        assert all(isinstance(region['media'], Audio) for region in regions)


def test_audio_segmenter_finds_moments(openai_backend, shared_speech):
    """Find the moment of a word in speech."""
    segmenter = AudioSegmenter(by='moment', prompt='the word "harvest" is spoken', backend=openai_backend)
    moments = segmenter.fit().predict([shared_speech])[0]
    assert moments
    _check_timed(moments, shared_speech.duration)


def test_video_segmenter_finds_moments(openai_backend, scenes_video):
    """Find the moment of the car in a video whose second shot shows it."""
    segmenter = VideoSegmenter(by='moment', prompt='a car is shown', n_frames=8, backend=openai_backend)
    moments = segmenter.fit().predict([scenes_video])[0]
    assert moments
    _check_timed(moments, scenes_video.duration)
    assert any(moment['end'] > scenes_video.duration / 2 for moment in moments)
    assert all(isinstance(moment['media'], Video) for moment in moments)


def test_video_thumbnail_is_a_frame(openai_backend, scenes_video):
    """Pick a thumbnail among the frames of a video."""
    thumbnail = VideoToImageGenerator(n_frames=4, backend=openai_backend).fit().predict([scenes_video])[0]
    assert isinstance(thumbnail, Image.Image)
    assert thumbnail.size == scenes_video.frames.shape[2:0:-1]


def test_video_is_reframed_to_portrait(openai_backend, apple_video):
    """Reframe a square video to 9:16 around the apple."""
    reframer = VideoToVideoGenerator(aspect_ratio=9 / 16, subject='apple', n_frames=2, backend=openai_backend)
    video = reframer.fit().predict([apple_video])[0]
    height, width = video.frames.shape[1:3]
    assert height / width == pytest.approx(16 / 9, rel=0.05)
    assert video.duration == pytest.approx(apple_video.duration, abs=0.1)


# Generation from multimodal samples


def test_multimodal_speech_uses_each_voice(openai_backend):
    """Speak the text of every sample in the voice of the same sample."""
    lines = [
        {'text': 'Hello, I am the first host.', 'voice': 'alloy'},
        {'text': 'And I am the second.', 'voice': 'echo'},
    ]
    speech = MultimodalToAudioGenerator(operation='speech', backend=openai_backend).fit().predict(lines)
    assert len(speech) == 2
    assert all(isinstance(audio, Audio) and audio.duration > 0.5 for audio in speech)
    transcript = AudioToTextGenerator(language='en', backend=openai_backend).fit().predict(speech[:1])[0].lower()
    assert 'host' in transcript


def test_multimodal_edit_follows_each_instruction(openai_backend, shared_images):
    """Edit the image of a sample with the instructions of the same sample."""
    car = shared_images[1]
    samples = [{'image': car, 'instructions': 'Make the car bright red.'}]
    edited = MultimodalToImageGenerator(backend=openai_backend).fit().predict(samples)[0]
    assert edited.size == car.size
    classifier = ImageClassifier(labels=['a red car', 'a blue car'], backend=openai_backend).fit()
    assert classifier.predict([edited]).tolist() == ['a red car']


def test_reference_prompt_keeps_the_subject(openai_backend):
    """Generate one reference image of a subject in fit and send it with every prompt."""
    generator = TextToImageGenerator(reference_prompt='A small friendly robot', backend=openai_backend).fit()
    assert len(generator.reference_images_) == 1
    (image,) = generator.predict(['The robot waters a plant'])
    assert image.size == (1024, 1024)
    description = ImageToTextGenerator(instructions='Describe the image.', backend=openai_backend).fit()
    assert 'robot' in description.predict([image])[0].lower()