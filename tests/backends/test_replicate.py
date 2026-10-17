"""Test the Replicate backend against a fake client."""

import numpy as np
import pytest
from PIL import Image

from skai.audio import TalkingAvatarGenerator
from skai.backends import ReplicateBackend
from skai.core import InvalidResponseError, SampleError, Video
from skai.text import ImageGenerator, SpeechSynthesizer
from tests.fakes import Recorder, install_module, namespace, png_of, wav_of

FAKE_CREDENTIAL = 'r8-test'


def file_output(data):
    """Return an object that reads like a Replicate file output."""
    return namespace(read=lambda: data)


def make_client(output):
    """Return a fake client whose models return `output`."""
    return namespace(run=Recorder(lambda model, **kwargs: output))


def test_connect_creates_client_from_environment(monkeypatch):
    """Create the client with the token read from `api_token_env`."""
    created = {}
    install_module(monkeypatch, 'replicate', Client=lambda **kwargs: created.update(kwargs) or make_client(None))
    monkeypatch.setenv('REPLICATE_API_TOKEN', FAKE_CREDENTIAL)
    ReplicateBackend().connect()
    assert created == {'api_token': FAKE_CREDENTIAL}


def test_avatar_uploads_portrait_and_audio(audio):
    """Upload the portrait and the audio, and return the encoded video."""
    data = Video(np.zeros((25, 8, 8, 3), dtype=np.uint8), fps=25).to_bytes()
    client = make_client(file_output(data))
    generator = TalkingAvatarGenerator(avatar=Image.new('RGB', (8, 8)), backend=ReplicateBackend(client=client)).fit()
    video = generator.predict([audio])[0]
    assert video.to_bytes() == data
    model, inputs = client.run.calls[-1]['args'][0], client.run.calls[-1]['input']
    assert model == 'cjwbw/sadtalker'
    assert (inputs['source_image'].name, inputs['driven_audio'].name) == ('avatar.png', 'audio.wav')


def test_avatar_is_the_default_backend(monkeypatch):
    """Use Replicate when `backend=None`."""
    install_module(monkeypatch, 'replicate', Client=lambda **kwargs: make_client(None))
    monkeypatch.setenv('REPLICATE_API_TOKEN', 'r8-test')
    generator = TalkingAvatarGenerator(avatar=Image.new('RGB', (8, 8))).fit()
    assert isinstance(generator.backend_, ReplicateBackend)


def test_image_generator_maps_size_to_aspect_ratio():
    """Send the closest aspect ratio and the seed, and resize to the requested size."""
    client = make_client([file_output(png_of((32, 18)))])
    generator = ImageGenerator(
        size=(64, 36),
        negative_prompt='text',
        random_state=0,
        backend=ReplicateBackend(client=client),
    )
    assert generator.fit().predict(['a cat'])[0].size == (64, 36)
    inputs = client.run.calls[-1]['input']
    assert inputs['aspect_ratio'] == '16:9'
    assert inputs['negative_prompt'] == 'text'
    assert 'seed' in inputs


def test_synthesizer_switches_to_cloning_model(audio):
    """Use the voice model for names and the cloning model for reference audio."""
    client = make_client(file_output(wav_of(0.25)))
    backend = ReplicateBackend(client=client, options={'speed': 1.2})
    assert SpeechSynthesizer(backend=backend).fit().predict(['hi'])[0].duration == pytest.approx(0.25)
    assert client.run.calls[-1]['args'][0] == 'jaaari/kokoro-82m'
    assert client.run.calls[-1]['input'] == {'text': 'hi', 'voice': 'af_bella', 'speed': 1.2}
    SpeechSynthesizer(voice=audio, language='el', backend=backend).fit().predict(['geia'])
    assert client.run.calls[-1]['args'][0] == 'lucataco/xtts-v2'
    assert client.run.calls[-1]['input']['language'] == 'el'


def test_url_output_is_downloaded(server):
    """Download outputs returned as URLs."""
    directory, url = server
    (directory / 'speech.wav').write_bytes(wav_of(0.5))
    client = make_client(f'{url}/speech.wav')
    speech = SpeechSynthesizer(backend=ReplicateBackend(client=client)).fit().predict(['hi'])[0]
    assert speech.duration == pytest.approx(0.5)


@pytest.mark.parametrize('output', [[], 42])
def test_unreadable_output_is_reported(output):
    """Report empty or unreadable outputs."""
    synthesizer = SpeechSynthesizer(backend=ReplicateBackend(client=make_client(output))).fit()
    with pytest.raises(SampleError) as error:
        synthesizer.predict(['hi'])
    assert isinstance(error.value.__cause__, InvalidResponseError)
