from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


TERMINAL_STATUSES = {"succeeded", "failed", "failed_manifest_sync", "cancelled"}


@dataclass(slots=True)
class IngestionJob:
    job_id: str
    workspace_id: str
    project: str
    status: str
    stage: str
    current: int
    total: int
    documents_total: int
    documents_succeeded: int
    documents_failed: int
    chunks_total: int
    current_file: str | None
    files_added: int
    files_updated: int
    files_skipped: int
    started_at: str
    updated_at: str
    finished_at: str | None = None
    active_index: str | None = None
    error: str | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @property
    def stage_percent(self) -> float | None:
        if self.total <= 0:
            return None
        return min(100.0, self.current / self.total * 100)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IngestionJob:
        data = dict(data)
        data.setdefault("current_file", None)
        data.setdefault("files_added", 0)
        data.setdefault("files_updated", 0)
        data.setdefault("files_skipped", 0)
        fields = cls.__dataclass_fields__
        return cls(**{key: value for key, value in data.items() if key in fields})
