from __future__ import annotations

import json
import sys
from typing import Any, TextIO


def emit_json(payload: dict[str, Any], *, stream: TextIO | None = None, indent: int | None = None) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=indent), file=stream or sys.stdout, flush=True)


def emit_error(code: str, message: str, **details: Any) -> int:
    error: dict[str, Any] = {"code": code, "message": message}
    if details:
        error["details"] = details
    emit_json({"ok": False, "error": error}, stream=sys.stderr)
    return 1
