from __future__ import annotations

import hashlib
import re
import time


def _short_hash(value: str) -> str:
    return hashlib.sha1(value.encode()).hexdigest()[:12]


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_excerpt(text: str, limit: int = 220) -> str:
    compact = re.sub(r"\s+", " ", (text or "").strip())
    return compact if len(compact) <= limit else compact[:limit] + "..."
