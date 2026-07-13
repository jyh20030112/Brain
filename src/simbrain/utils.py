from __future__ import annotations

import hashlib
from datetime import datetime, timezone


def _short_hash(value: str) -> str:
    return hashlib.sha1(value.encode()).hexdigest()[:12]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
