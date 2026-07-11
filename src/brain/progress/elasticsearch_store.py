from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from brain.progress.models import IngestionJob
from brain.storage.client import es_context, response_body, run_async


PROGRESS_INDEX = "brain_ingestion_jobs"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ElasticsearchProgressStore:
    """将跨进程可见的入库任务进度保存到独立 ES 索引。"""

    def __init__(
        self,
        *,
        es_url: str = "",
        es_cloud_id: str = "",
        es_user: str = "",
        es_pass: str = "",
        es_api_key: str = "",
    ):
        self.es_url = es_url
        self.es_cloud_id = es_cloud_id
        self.es_user = es_user
        self.es_pass = es_pass
        self.es_api_key = es_api_key

    @staticmethod
    def _mapping() -> dict:
        return {
            "properties": {
                "job_id": {"type": "keyword"},
                "workspace_id": {"type": "keyword"},
                "project": {"type": "keyword"},
                "status": {"type": "keyword"},
                "stage": {"type": "keyword"},
                "current": {"type": "long"},
                "total": {"type": "long"},
                "documents_total": {"type": "long"},
                "documents_succeeded": {"type": "long"},
                "documents_failed": {"type": "long"},
                "chunks_total": {"type": "long"},
                "started_at": {"type": "date"},
                "updated_at": {"type": "date"},
                "finished_at": {"type": "date"},
                "active_index": {"type": "keyword"},
                "error": {"type": "text", "index": False},
            }
        }

    def _context(self):
        return es_context(
            url=self.es_url,
            cloud_id=self.es_cloud_id,
            api_key=self.es_api_key,
            username=self.es_user,
            password=self.es_pass,
        )

    async def _ensure_index(self, es) -> None:
        if not await es.indices.exists(index=PROGRESS_INDEX):
            await es.options(ignore_status=400).indices.create(index=PROGRESS_INDEX, mappings=self._mapping())

    def create_job(self, *, project: str, workspace_id: str) -> IngestionJob:
        async def _run():
            async with self._context() as es:
                await self._ensure_index(es)
                now = _utc_now()
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
                    started_at=now,
                    updated_at=now,
                )
                await es.index(index=PROGRESS_INDEX, id=job.job_id, document=job.to_dict(), refresh="wait_for")
                return job

        return run_async(_run())

    def update_job(self, job_id: str, **fields) -> None:
        async def _run():
            async with self._context() as es:
                fields["updated_at"] = _utc_now()
                await es.update(index=PROGRESS_INDEX, id=job_id, doc=fields, refresh="wait_for")

        run_async(_run())

    def complete_job(self, job_id: str, *, active_index: str) -> None:
        now = _utc_now()
        self.update_job(
            job_id,
            status="succeeded",
            stage="completed",
            current=1,
            total=1,
            active_index=active_index,
            finished_at=now,
            error=None,
        )

    def fail_job(self, job_id: str, *, error: str) -> None:
        self.update_job(
            job_id,
            status="failed",
            finished_at=_utc_now(),
            error=error[:2000],
        )

    def get_job(self, job_id: str) -> IngestionJob | None:
        async def _run():
            async with self._context() as es:
                if not await es.indices.exists(index=PROGRESS_INDEX):
                    return None
                response = response_body(await es.options(ignore_status=404).get(index=PROGRESS_INDEX, id=job_id))
                if not response.get("found", True) or "_source" not in response:
                    return None
                return IngestionJob.from_dict(response["_source"])

        return run_async(_run())

    def list_jobs(self, *, workspace_id: str, limit: int = 10) -> list[IngestionJob]:
        async def _run():
            async with self._context() as es:
                if not await es.indices.exists(index=PROGRESS_INDEX):
                    return []
                response = response_body(
                    await es.search(
                        index=PROGRESS_INDEX,
                        size=limit,
                        query={"term": {"workspace_id": workspace_id}},
                        sort=[{"started_at": {"order": "desc"}}],
                    )
                )
                return [IngestionJob.from_dict(hit["_source"]) for hit in response.get("hits", {}).get("hits", [])]

        return run_async(_run())
