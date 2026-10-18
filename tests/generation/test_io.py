"""Test the sample containers and output formats of the generators."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from skai.core import Audio, read_video
from skai.generation import (
    AudioToTextGenerator,
    AudioToVideoGenerator,
    ImageToTextGenerator,
    TextToAudioGenerator,
    TextToImageGenerator,
    TextToTextGenerator,
    TextToVideoGenerator,
)


def test_dataframe_column_input():
    """Return one summary per row of a one-column DataFrame."""
    frame = pd.DataFrame({'text': [f'Text number {index}.' for index in range(5)]})
    summaries = TextToTextGenerator(instructions='Summarize the text.', backend='dummy').fit().predict(frame)
    assert summaries.shape == (5,)


def test_multi_column_input_rejected():
    """Explain that one column of samples is expected."""
    frame = pd.DataFrame({'a': ['x', 'y'], 'b': ['z', 'w']})
    with pytest.raises(ValueError, match='one column'):
        TextToTextGenerator(instructions='Summarize the text.', backend='dummy').fit().predict(frame)


def test_mixed_image_forms(server, png_bytes, image):
    """Describe an image given as a path, a URL and a PIL image."""
    directory, url = server
    (directory / 'photo.png').write_bytes(png_bytes)
    generator = ImageToTextGenerator(instructions='Describe the image.', backend='dummy').fit()
    descriptions = generator.predict([directory / 'photo.png', f'{url}/photo.png', image])
    assert descriptions.tolist() == ['A 32x24 image.'] * 3


def test_array_audio_input():
    """Transcribe a `(samples, sample_rate)` tuple."""
    assert AudioToTextGenerator(backend='dummy').fit().predict([(np.zeros(16000), 16000)]).shape == (1,)


def test_text_output_type():
    """Return a 1-D object array of strings."""
    outputs = TextToTextGenerator(instructions='Translate into French.', backend='dummy').fit().predict(list('abcd'))
    assert outputs.shape == (4,)
    assert outputs.dtype == object
    assert all(isinstance(output, str) for output in outputs)


def test_in_memory_image_output():
    """Return PIL images with `output_format='object'`."""
    (output,) = TextToImageGenerator(size=(8, 8), output_format='object', backend='dummy').fit().predict(['a cat'])
    assert isinstance(output, Image.Image)


def test_bytes_output():
    """Return WAV bytes for audio, PNG bytes for images and MP4 bytes for videos."""
    (audio,) = TextToAudioGenerator(output_format='bytes', backend='dummy').fit().predict(['hi'])
    assert isinstance(audio, bytes)
    assert audio[:4] == b'RIFF'
    (image,) = TextToImageGenerator(size=(8, 8), output_format='bytes', backend='dummy').fit().predict(['a cat'])
    assert image[:8] == b'\x89PNG\r\n\x1a\n'
    generator = TextToVideoGenerator(duration=1, size=(16, 16), output_format='bytes', backend='dummy')
    (video,) = generator.fit().predict(['a boat'])
    assert b'ftyp' in video[:16]


def test_paths_into_a_directory(tmp_path):
    """Write new files under `output_dir`, return their paths in input order, and never overwrite files."""
    output_dir = tmp_path / 'out'
    portrait = Image.new('RGB', (16, 16))
    speech = [Audio(np.zeros(8000 * (index + 1), dtype=np.float32), 16000) for index in range(2)]
    generator = AudioToVideoGenerator(avatar=portrait, output_format='path', output_dir=output_dir, backend='dummy')
    paths = generator.fit().predict(speech)
    assert len(paths) == 2
    assert len(set(paths)) == 2
    assert sorted(Path(path).name for path in paths) == sorted(path.name for path in output_dir.iterdir())
    assert [read_video(path).duration for path in paths] == pytest.approx([0.5, 1.0], abs=0.1)
    before = {path: Path(path).read_bytes() for path in paths}
    again = generator.predict(speech)
    assert not set(again) & set(paths)
    assert all(Path(path).read_bytes() == content for path, content in before.items())


def test_temporary_directory_without_output_dir():
    """Write path outputs to a temporary directory when `output_dir` is `None`."""
    (path,) = TextToAudioGenerator(output_format='path', backend='dummy').fit().predict(['hi'])
    assert Path(path).is_file()
    assert str(path).endswith('.wav')


def test_round_trip_between_components():
    """Transcribe the in-memory audio of the speech generator."""
    speech = TextToAudioGenerator(backend='dummy').fit().predict(['Hello there'])
    assert isinstance(speech[0], Audio)
    assert AudioToTextGenerator(backend='dummy').fit().predict(speech)[0].startswith('Speech of')
