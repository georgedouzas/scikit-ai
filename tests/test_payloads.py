"""Check that recorded provider payloads hold no credentials."""

import re
from pathlib import Path

PAYLOADS = Path(__file__).parent / 'payloads'
SECRET = re.compile(r'(sk-[A-Za-z0-9_-]{16,}|r8_[A-Za-z0-9]{16,}|Bearer\s+\S+|api[_-]?key)', re.IGNORECASE)


def test_recordings_hold_no_credentials():
    """Find nothing shaped like an API key or an authorization header in any recording."""
    leaks = [path.name for path in PAYLOADS.rglob('*.json') if SECRET.search(path.read_text())]
    assert not leaks, f'Recordings that look like they hold credentials: {leaks}'
