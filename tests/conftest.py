"""Configure the test suite: block the network and provide small media samples."""

import io
import socket
from collections.abc import Iterator
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any

import numpy as np
import pytest
from PIL import Image

from skai.core import Audio

LOCAL_HOSTS = {'127.0.0.1', 'localhost', '::1'}
SAMPLE_RATE = 16000


class QuietHandler(SimpleHTTPRequestHandler):
    """Serve files without logging every request."""

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Skip the request log."""


@pytest.fixture(autouse=True)
def block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any test that resolves or connects to a host other than the local machine."""
    connect = socket.socket.connect
    getaddrinfo = socket.getaddrinfo

    def guarded_connect(self: socket.socket, address: Any) -> None:
        """Connect only to local addresses."""
        host = address[0] if isinstance(address, tuple) else address
        if host not in LOCAL_HOSTS:
            msg = f'Tests must not reach the network, but connected to {address}.'
            raise RuntimeError(msg)
        connect(self, address)

    def guarded_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        """Resolve only local host names."""
        if host not in LOCAL_HOSTS:
            msg = f'Tests must not reach the network, but resolved {host}.'
            raise RuntimeError(msg)
        return getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, 'connect', guarded_connect)
    monkeypatch.setattr(socket, 'getaddrinfo', guarded_getaddrinfo)


@pytest.fixture
def server(tmp_path: Path) -> Iterator[tuple[Path, str]]:
    """Serve the files of a temporary directory over HTTP on the local machine."""
    httpd = ThreadingHTTPServer(('127.0.0.1', 0), partial(QuietHandler, directory=str(tmp_path)))
    thread = Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield tmp_path, f'http://127.0.0.1:{httpd.server_port}'
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture
def image() -> Image.Image:
    """Return a small RGB image."""
    return Image.new('RGB', (32, 24), (200, 30, 30))


@pytest.fixture
def png_bytes(image: Image.Image) -> bytes:
    """Return the PNG bytes of the small image."""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return buffer.getvalue()


@pytest.fixture
def audio() -> Audio:
    """Return one second of a quiet tone."""
    times = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    return Audio((0.1 * np.sin(2 * np.pi * 440 * times)).astype(np.float32), SAMPLE_RATE)
