import json

from brain.cli import ingest, search, status
from brain.config import Config
from brain.models import RetrievedChunk, TextChunk
from brain.progress.models import IngestionJob


def _result() -> RetrievedChunk:
    chunk = TextChunk(
        id="chunk_1",
        workspace_id="wid",
        file_name="manual.md",
        source_path="/docs/manual.md",
        content="早晚使用。",
        page_number=2,
        section="使用方法",
        chunk_type="paragraph",
    )
    return RetrievedChunk(chunk=chunk, score=0.03, retrieval_method="rrf")


def test_search_cli_outputs_machine_readable_json(monkeypatch, capsys):
    class FakeService:
        def search(self, query, *, top_k):
            assert (query, top_k) == ("怎么使用？", 3)
            return [_result()]

    monkeypatch.setattr(search.SearchService, "from_config", lambda cfg: FakeService())

    assert search.main(["怎么使用？", "--top-k", "3", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["results"][0]["chunk"]["content"] == "早晚使用。"


def test_ingest_cli_applies_command_line_overrides(monkeypatch):
    captured = {}
    monkeypatch.setattr(ingest, "run_ingestion", lambda cfg: captured.update(cfg=cfg))

    assert ingest.main(["--input-dir", "docs", "--project", "my-knowledge-base"]) == 0
    assert captured["cfg"].input_dir == "docs"
    assert captured["cfg"].project == "my-knowledge-base"


def _job(*, job_status="running", stage="embedding", current=2, total=4):
    return IngestionJob(
        job_id="ingest_test",
        workspace_id="wid",
        project="my-knowledge-base",
        status=job_status,
        stage=stage,
        current=current,
        total=total,
        documents_total=2,
        documents_succeeded=2,
        documents_failed=0,
        chunks_total=4,
        started_at="2026-07-12T00:00:00+00:00",
        updated_at="2026-07-12T00:00:01+00:00",
    )


def _status_config():
    return Config(project="my-knowledge-base", es_url="http://fake-es")


def test_status_cli_outputs_latest_job_as_json(monkeypatch, capsys):
    class FakeProgressStore:
        def list_jobs(self, *, workspace_id, limit):
            return [_job(job_status="succeeded", stage="completed", current=1, total=1)]

    monkeypatch.setattr(status.Config, "from_env", classmethod(lambda cls: _status_config()))
    monkeypatch.setattr(status, "build_progress_store", lambda cfg: FakeProgressStore())

    assert status.main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["job_id"] == "ingest_test"
    assert payload["status"] == "succeeded"


def test_status_cli_watch_refreshes_until_terminal(monkeypatch, capsys):
    store = SequenceStatusStore(
        [
            _job(),
            _job(job_status="succeeded", stage="completed", current=1, total=1),
        ]
    )
    monkeypatch.setattr(status.Config, "from_env", classmethod(lambda cls: _status_config()))
    monkeypatch.setattr(status, "build_progress_store", lambda cfg: store)
    monkeypatch.setattr(status.time, "sleep", lambda interval: None)

    assert status.main(["--job-id", "ingest_test", "--watch", "--json"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert [json.loads(line)["status"] for line in lines] == ["running", "succeeded"]


class SequenceStatusStore:
    def __init__(self, jobs):
        self.jobs = jobs

    def get_job(self, job_id):
        return self.jobs.pop(0)
