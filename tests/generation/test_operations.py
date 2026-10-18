"""Test what every media generator returns and how operations are chosen and validated."""

import sys

import numpy as np
import pytest
from PIL import Image
from sklearn.base import clone
from sklearn.utils._param_validation import InvalidParameterError

import skai.generation
from skai.backends import OpenAIBackend
from skai.core import Audio, InvalidSampleError, UnsupportedBackendError, Video
from skai.generation import (
    AudioToAudioGenerator,
    AudioToVideoGenerator,
    ImageToImageGenerator,
    ImageToVideoGenerator,
    MultimodalToAudioGenerator,
    MultimodalToImageGenerator,
    MultimodalToVideoGenerator,
    TextToAudioGenerator,
    TextToImageGenerator,
    TextToVideoGenerator,
    VideoToAudioGenerator,
    VideoToImageGenerator,
    VideoToVideoGenerator,
)
from tests.fakes import namespace

IMAGE = Image.new('RGB', (16, 12), 'teal')
AUDIO = Audio(np.zeros(16000, dtype=np.float32), 16000)
VIDEO = Video(np.zeros((8, 12, 16, 3), dtype=np.uint8), fps=8, audio=AUDIO)
OPERATIONS = {
    TextToImageGenerator: ['generate'],
    ImageToImageGenerator: ['edit', 'upscale', 'remove_background', 'collage'],
    VideoToImageGenerator: ['thumbnail', 'frame'],
    MultimodalToImageGenerator: ['edit'],
    TextToAudioGenerator: ['speech', 'music', 'sound'],
    AudioToAudioGenerator: ['separate', 'enhance', 'convert_voice', 'concatenate'],
    VideoToAudioGenerator: ['extract'],
    MultimodalToAudioGenerator: ['mix', 'speech'],
    TextToVideoGenerator: ['generate'],
    ImageToVideoGenerator: ['animate'],
    AudioToVideoGenerator: ['talking_head'],
    VideoToVideoGenerator: ['reframe', 'concatenate'],
    MultimodalToVideoGenerator: ['lip_sync', 'mux', 'subtitle'],
}
CASES = [
    (TextToImageGenerator(size=(8, 8), backend='dummy'), 'a cat', Image.Image),
    (ImageToImageGenerator(instructions='add snow', backend='dummy'), IMAGE, Image.Image),
    (ImageToImageGenerator(operation='upscale', backend='dummy'), IMAGE, Image.Image),
    (ImageToImageGenerator(operation='remove_background', backend='dummy'), IMAGE, Image.Image),
    (ImageToImageGenerator(operation='collage', tile=8), [IMAGE, IMAGE], Image.Image),
    (VideoToImageGenerator(backend='dummy'), VIDEO, Image.Image),
    (VideoToImageGenerator(operation='frame', at=0.5), VIDEO, Image.Image),
    (TextToAudioGenerator(backend='dummy'), 'hello', Audio),
    (TextToAudioGenerator(operation='music', duration=1, backend='dummy'), 'calm piano', Audio),
    (TextToAudioGenerator(operation='sound', duration=1, backend='dummy'), 'rain', Audio),
    (AudioToAudioGenerator(backend='dummy'), AUDIO, Audio),
    (AudioToAudioGenerator(operation='enhance', backend='dummy'), AUDIO, Audio),
    (AudioToAudioGenerator(operation='convert_voice', voice=AUDIO, backend='dummy'), AUDIO, Audio),
    (AudioToAudioGenerator(operation='concatenate'), [AUDIO, AUDIO], Audio),
    (VideoToAudioGenerator(), VIDEO, Audio),
    (MultimodalToAudioGenerator(), {'voice': AUDIO, 'music': AUDIO}, Audio),
    (MultimodalToAudioGenerator(operation='speech', backend='dummy'), {'text': 'Hello', 'voice': 'nova'}, Audio),
    (MultimodalToImageGenerator(backend='dummy'), {'image': IMAGE, 'instructions': 'add snow'}, Image.Image),
    (TextToVideoGenerator(duration=1, size=(16, 12), backend='dummy'), 'a boat', Video),
    (ImageToVideoGenerator(duration=1, backend='dummy'), IMAGE, Video),
    (AudioToVideoGenerator(avatar=IMAGE, backend='dummy'), AUDIO, Video),
    (VideoToVideoGenerator(backend='dummy'), VIDEO, Video),
    (VideoToVideoGenerator(operation='concatenate'), [VIDEO, VIDEO], Video),
    (MultimodalToVideoGenerator(backend='dummy'), {'video': VIDEO, 'audio': AUDIO}, Video),
    (MultimodalToVideoGenerator(operation='mux'), {'video': VIDEO, 'audio': AUDIO}, Video),
    (MultimodalToVideoGenerator(operation='subtitle'), {'video': VIDEO, 'transcript': 'Hello.'}, Video),
]


@pytest.mark.parametrize(('generator', 'sample', 'output_type'), CASES)
def test_output_type_follows_the_name(generator, sample, output_type):
    """Return PIL images, `Audio` or `Video` as the class name ends, and equal outputs from predict and transform."""
    suffix = {Image.Image: 'ImageGenerator', Audio: 'AudioGenerator', Video: 'VideoGenerator'}[output_type]
    assert type(generator).__name__.endswith(suffix)
    outputs = clone(generator).fit().predict([sample])
    assert outputs.shape == (1,)
    assert outputs.dtype == object
    assert isinstance(outputs[0], output_type)


def test_every_generator_is_covered():
    """Cover every media generator and every operation in the output tests."""
    covered = {(type(generator), generator.operation) for generator, _, _ in CASES}
    assert covered == {(cls, operation) for cls, operations in OPERATIONS.items() for operation in operations}
    media = [name for name in skai.generation.__all__ if not name.endswith('ToTextGenerator')]
    assert sorted(cls.__name__ for cls in OPERATIONS) == sorted(media)


@pytest.mark.parametrize(('generator_class', 'operations'), OPERATIONS.items())
def test_operations_per_generator(generator_class, operations):
    """Declare the operations of every generator, default first."""
    assert list(generator_class.operations) == operations
    assert generator_class().operation == operations[0]


def test_invalid_operation():
    """List the operations of the generator when the operation is invalid."""
    with pytest.raises(InvalidParameterError, match='operation') as info:
        TextToAudioGenerator(operation='edit', backend='dummy').fit()
    assert all(operation in str(info.value) for operation in OPERATIONS[TextToAudioGenerator])


@pytest.mark.parametrize('generator_class', list(OPERATIONS))
def test_invalid_operation_on_every_generator(generator_class):
    """Validate the operation of every generator at `fit`."""
    params = {'avatar': IMAGE} if generator_class is AudioToVideoGenerator else {}
    with pytest.raises(InvalidParameterError, match="'operation' parameter"):
        generator_class(operation='unknown', **params).fit()


def test_unsupported_operation():
    """Raise at `fit`, without a client, when the backend lacks the operation, and list the backends that have it."""
    generator = MultimodalToVideoGenerator(operation='lip_sync', backend='openai')
    with pytest.raises(UnsupportedBackendError, match=r"does not support `lip_sync`.*\['dummy', 'replicate'\]"):
        generator.fit()
    assert not hasattr(generator, 'backend_') or generator.backend_ is None
    with pytest.raises(UnsupportedBackendError, match='convert_voice'):
        AudioToAudioGenerator(operation='convert_voice', voice=AUDIO, backend=OpenAIBackend(client=namespace())).fit()
    with pytest.raises(UnsupportedBackendError, match='generate_music'):
        TextToAudioGenerator(operation='music', backend=OpenAIBackend(client=namespace())).fit()


@pytest.mark.parametrize(
    ('generator', 'sample'),
    [
        (ImageToImageGenerator(operation='collage', tile=8), [IMAGE]),
        (VideoToImageGenerator(operation='frame'), VIDEO),
        (AudioToAudioGenerator(operation='concatenate'), [AUDIO]),
        (VideoToAudioGenerator(), VIDEO),
        (MultimodalToAudioGenerator(), {'a': AUDIO}),
        (VideoToVideoGenerator(operation='concatenate', backend=None), [VIDEO]),
        (MultimodalToVideoGenerator(operation='mux', backend=None), {'video': VIDEO, 'audio': AUDIO}),
        (MultimodalToVideoGenerator(operation='subtitle', backend=None), {'video': VIDEO, 'transcript': 'Hi.'}),
    ],
)
def test_local_operations_need_no_backend(generator, sample):
    """Run local operations without a backend."""
    fitted = clone(generator).fit()
    assert fitted.backend_ is None
    assert fitted.predict([sample])[0] is not None


def test_offline_assembly():
    """Join three clips into one video of the summed durations without importing any provider."""
    providers = {'openai', 'transformers', 'replicate', 'litellm', 'torch'}
    before = set(sys.modules)
    clips = [Video(np.zeros((count, 4, 4, 3), dtype=np.uint8), fps=4) for count in (4, 8, 2)]
    generator = VideoToVideoGenerator(operation='concatenate', backend=None).fit()
    assert generator.backend_ is None
    assert generator.predict([clips])[0].duration == 3.5
    new = set(sys.modules) - before
    assert not [name for name in new if name.split('.')[0] in providers]


def test_list_samples_need_a_list_operation():
    """Reject a list sample outside list operations, suggesting `ForEach`."""
    with pytest.raises(InvalidSampleError, match=r'Sample 0 is a list.*ForEach'):
        ImageToImageGenerator(operation='upscale', backend='dummy').fit().predict([[IMAGE, IMAGE]])
    with pytest.raises(InvalidSampleError, match='ForEach'):
        TextToImageGenerator(size=(4, 4), backend='dummy').fit().predict([['a', 'b']])
