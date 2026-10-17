"""Check that the OpenAI adapter still sends the requests that the recorded real runs sent."""

import json
from pathlib import Path

from PIL import Image

from skai.backends import OpenAIBackend
from skai.description import ImageDescriptor
from tests.backends.test_openai import make_client

RECORDINGS = Path(__file__).parents[1] / 'payloads' / 'openai'


def load_request(name):
    """Return the recorded request body of a recording."""
    return json.loads((RECORDINGS / name).read_text())['request']


def test_descriptor_examples_match_recorded_request():
    """Send the example image and the target image with the same text as the recorded real request."""
    recorded = load_request('test_descriptor_follows_reference_style-1.json')
    client = make_client('Object: car. Color: blue.')
    images = [Image.new('RGB', (8, 8), 'red'), Image.new('RGB', (8, 8), 'blue')]
    descriptor = ImageDescriptor(backend=OpenAIBackend(client=client)).fit(images[:1], ['Object: apple. Color: red.'])
    descriptor.predict(images[1:])
    sent = client.chat.completions.create.calls[-1]
    recorded_parts = recorded['messages'][1]['content']
    sent_parts = sent['messages'][1]['content']
    assert [part['type'] for part in sent_parts] == [part['type'] for part in recorded_parts]
    assert sent_parts[0]['text'] == recorded_parts[0]['text']
    assert sent['messages'][0] == recorded['messages'][0]
    assert sent['model'] == recorded['model']
