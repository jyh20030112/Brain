from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
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
    """同一主机上按 workspace 加锁，同时保留 project 目录锁文件。"""

    def __init__(self, workspace_id: str, project_dir: Path, *, lock_root: Path | None = None):
        root = lock_root or Path(tempfile.gettempdir()) / "brain-ingest-locks"
        self.path = root / f"{workspace_id}.lock"
        self.project_path = project_dir / ".ingest.lock"
        self._global_lock = FileLock(str(self.path))
        self._project_lock = FileLock(str(self.project_path))

    def __enter__(self) -> ProjectLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.project_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._global_lock.acquire(timeout=0)
            self._project_lock.acquire(timeout=0)
        except Timeout as exc:
            if self._global_lock.is_locked:
                self._global_lock.release()
            raise ProjectLockedError("该 project 已有入库任务正在运行") from exc
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._project_lock.release()
        self._global_lock.release()
