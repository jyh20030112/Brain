from __future__ import annotations

import sys
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from simbrain.progress.models import IngestionJob
from simbrain.project import atomic_write_json, read_json
from simbrain.utils import utc_now


class FileProgressStore:
    """将当前或最近一次任务状态原子写入 project/progress.json。"""

    def __init__(self, project_dir: Path):
        self.path = project_dir / "progress.json"
        self._mutex = threading.RLock()

    def create_job(self, *, project: str, workspace_id: str) -> IngestionJob:
        now = utc_now()
        job = IngestionJob(
            job_id=f"ingest_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid4().hex[:8]}",
            workspace_id=workspace_id,
            project=project,
            status="running",
            stage="scanning",
            current=0,
            total=0,
            documents_total=0,
            documents_succeeded=0,
            documents_failed=0,
            chunks_total=0,
            current_file=None,
            files_added=0,
            files_updated=0,
            files_skipped=0,
            started_at=now,
            updated_at=now,
        )
        with self._mutex:
            atomic_write_json(self.path, job.to_dict())
        return job

    def update_job(self, job_id: str, **fields) -> None:
        with self._mutex:
            payload = read_json(self.path)
            if payload is None or payload.get("job_id") != job_id:
                raise RuntimeError(f"进度任务不存在或已被替换: {job_id}")
            fields["updated_at"] = utc_now()
            payload.update(fields)
            atomic_write_json(self.path, payload)

    def complete_job(self, job_id: str, *, active_index: str) -> None:
        self.update_job(
            job_id,
            status="succeeded",
            stage="completed",
            current=1,
            total=1,
            current_file=None,
            active_index=active_index,
            finished_at=utc_now(),
            error=None,
        )

    def fail_job(self, job_id: str, *, error: str, status: str = "failed") -> None:
        self.update_job(
            job_id,
            status=status,
            current_file=None,
            finished_at=utc_now(),
            error=error[:2000],
        )

    def get_job(self) -> IngestionJob | None:
        with self._mutex:
            payload = read_json(self.path)
        return IngestionJob.from_dict(payload) if payload else None

    @contextmanager
    def heartbeat(self, job_id: str, *, interval: float = 5.0):
        stop = threading.Event()

        def _run() -> None:
            consecutive_failures = 0
            while not stop.wait(interval):
                try:
                    job = self.get_job()
                    if job is None or job.job_id != job_id or job.is_terminal:
                        return
                    self.update_job(job_id)
                    consecutive_failures = 0
                except Exception as exc:
                    consecutive_failures += 1
                    print(
                        f"[进度] 心跳写入失败（连续 {consecutive_failures} 次）: {exc}",
                        file=sys.stderr,
                    )

        thread = threading.Thread(target=_run, name=f"heartbeat-{job_id}", daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=max(interval, 1.0))
