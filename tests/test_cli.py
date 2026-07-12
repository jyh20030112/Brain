import json

import pytest

from brain.cli import ingest, search, status
from brain.models import RetrievalOutcome, RetrievedChunk, TextChunk
from brain.progress.models import IngestionJob
from brain.project import atomic_write_json


def _result() -> RetrievedChunk:
    chunk = TextChunk(
        id="chunk_1",
        workspace_id="wid",
        file_name="manual.md",
        source_path="manual.md",
        content="按照说明配置。",
        page_number=2,
        section="配置方法",
        chunk_type="paragraph",
    )
    return RetrievedChunk(chunk=chunk, score=0.03, retrieval_method="rrf")


def test_ingest_cli_requires_three_arguments_and_outputs_json(monkeypatch, capsys):
    captured = {}

    def fake_run(cfg):
        captured["cfg"] = cfg
        return {"ok": True, "project": cfg.project, "added": 1}

    monkeypatch.setattr(ingest, "run_ingestion", fake_run)
    result = ingest.main(
        ["--input-dir", "docs", "--output-dir", "output", "--project", "my-knowledge-base"]
    )

    assert result == 0
    assert json.loads(capsys.readouterr().out)["added"] == 1
    assert captured["cfg"].input_dir == "docs"


def test_search_cli_requires_fixed_flags_and_outputs_json(monkeypatch, capsys):
    class FakeService:
        def search(self, question, *, top_k):
            assert (question, top_k) == ("如何配置？", 10)
            return RetrievalOutcome(results=[_result()], warnings=[])

    monkeypatch.setattr(search.SearchService, "from_config", lambda cfg: FakeService())
    assert search.main(
        ["--question", "如何配置？", "--project", "my-knowledge-base", "--top-k", "10"]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["project"] == "my-knowledge-base"
    assert payload["count"] == 1


@pytest.mark.parametrize("module", [ingest, search])
def test_cli_missing_required_arguments_is_rejected(module):
    with pytest.raises(SystemExit):
        module.main([])


def test_status_catalog_lists_projects_and_keeps_invalid_manifest(monkeypatch, tmp_path, capsys):
    output_dir = tmp_path / "output"
    valid = output_dir / "alpha"
    invalid = output_dir / "broken"
    atomic_write_json(
        valid / "manifest.json",
        {
            "project": "alpha",
            "description": "包含 1 份资料。",
            "topics": [],
            "file_count": 1,
            "chunk_count": 2,
            "updated_at": "now",
            "active_index": "docs_alpha_current",
            "files": [{"file_name": "a.txt"}],
        },
    )
    invalid.mkdir(parents=True)
    (invalid / "manifest.json").write_text("{bad", encoding="utf-8")

    assert status.main(["--output-dir", str(output_dir)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["count"] == 2
    assert payload["projects"][0]["project"] == "alpha"
    assert "error" in payload["projects"][1]


def _job(state="running", stage="embedding", current=1, total=2):
    return IngestionJob(
        job_id="ingest_test",
        workspace_id="wid",
        project="my-knowledge-base",
        status=state,
        stage=stage,
        current=current,
        total=total,
        documents_total=1,
        documents_succeeded=1,
        documents_failed=0,
        chunks_total=2,
        current_file=None,
        files_added=1,
        files_updated=0,
        files_skipped=0,
        started_at="2099-01-01T00:00:00+00:00",
        updated_at="2099-01-01T00:00:01+00:00",
    )


def test_status_project_mode_streams_json_lines_until_terminal(monkeypatch, tmp_path, capsys):
    class SequenceStore:
        def __init__(self, project_dir):
            self.jobs = [_job(), _job("succeeded", "completed", 1, 1)]

        def get_job(self):
            return self.jobs.pop(0)

    monkeypatch.setattr(status, "FileProgressStore", SequenceStore)
    monkeypatch.setattr(status.time, "sleep", lambda seconds: None)

    assert status.main(["--output-dir", str(tmp_path), "--project", "my-knowledge-base"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert [json.loads(line)["status"] for line in lines] == ["running", "succeeded"]


def test_status_does_not_emit_again_when_persisted_state_is_unchanged(monkeypatch, tmp_path, capsys):
    running = _job()

    class SequenceStore:
        def __init__(self, project_dir):
            self.jobs = [running, running, _job("succeeded", "completed", 1, 1)]

        def get_job(self):
            return self.jobs.pop(0)

    monkeypatch.setattr(status, "FileProgressStore", SequenceStore)
    monkeypatch.setattr(status.time, "sleep", lambda seconds: None)

    assert status.main(["--output-dir", str(tmp_path), "--project", "my-knowledge-base"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()

    assert [json.loads(line)["status"] for line in lines] == ["running", "succeeded"]


def test_status_project_mode_exits_nonzero_for_stale_job(monkeypatch, tmp_path, capsys):
    stale = _job()
    stale.updated_at = "2000-01-01T00:00:00+00:00"

    class StaleStore:
        def __init__(self, project_dir):
            pass

        def get_job(self):
            return stale

    monkeypatch.setattr(status, "FileProgressStore", StaleStore)

    assert status.main(["--output-dir", str(tmp_path), "--project", "my-knowledge-base"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "stale"
    assert payload["ok"] is False
    assert payload["error"]["code"] == "ingestion_stale"
