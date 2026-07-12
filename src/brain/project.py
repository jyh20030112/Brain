from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout


_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


class ProjectLockedError(RuntimeError):
    pass


def validate_project_name(project: str) -> str:
    value = project.strip()
    if not value:
        raise ValueError("project 不能为空")
    if value in {".", ".."}:
        raise ValueError("project 不能是 . 或 ..")
    if len(value) > 128:
        raise ValueError("project 长度不能超过 128 个字符")
    if "/" in value or "\\" in value or _CONTROL_CHARS.search(value):
        raise ValueError("project 不能包含路径分隔符或控制字符")
    return value


def get_project_dir(output_dir: str | Path, project: str) -> Path:
    root = Path(output_dir).expanduser().resolve()
    return root / validate_project_name(project)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON 根节点必须是对象: {path}")
    return data


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ProjectLock:
    def __init__(self, project_dir: Path):
        self.path = project_dir / ".ingest.lock"
        self._lock = FileLock(str(self.path))

    def __enter__(self) -> ProjectLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._lock.acquire(timeout=0)
        except Timeout as exc:
            raise ProjectLockedError("该 project 已有入库任务正在运行") from exc
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._lock.release()
