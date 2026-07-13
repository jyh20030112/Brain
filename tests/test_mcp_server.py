import asyncio
from pathlib import Path
from typing import cast

from fastmcp import Client, Context

from brain.progress.models import IngestionJob
from brain.project import atomic_write_json
from brain.serve import __main__ as serve_main
from brain.serve.server import brain_status, brain_status_realtime, mcp


def test_mcp_server_exposes_four_brain_tools():
    async def list_tools() -> set[str]:
        async with Client(mcp) as client:
            return {tool.name for tool in await client.list_tools()}

    assert asyncio.run(list_tools()) == {
        "brain-ingest",
        "brain-status",
        "brain-status-realtime",
        "brain-search",
    }


def test_mcp_entrypoint_uses_configured_http_transport(monkeypatch):
    captured: dict[str, object] = {}
    monkeypatch.setattr(serve_main.mcp, "run", lambda **kwargs: captured.update(kwargs))

    serve_main.main()

    assert captured == {"transport": "http", "host": "0.0.0.0", "port": 2418, "path": "/mcp"}


def test_status_tool_returns_the_cli_catalog_payload(tmp_path: Path):
    output_dir = tmp_path / "output"
    atomic_write_json(output_dir / "alpha" / "manifest.json", {"project": "alpha", "files": []})

    payload = brain_status(str(output_dir))

    assert payload["ok"] is True
    assert payload["count"] == 1
    assert payload["projects"][0]["project"] == "alpha"


def test_realtime_status_tool_reports_and_returns_terminal_event(tmp_path: Path):
    class FakeContext:
        def __init__(self) -> None:
            self.messages: list[str] = []
            self.progress: list[tuple[float, int]] = []

        async def info(self, message: str) -> None:
            self.messages.append(message)

        async def report_progress(self, *, progress: float, total: int) -> None:
            self.progress.append((progress, total))

    job = IngestionJob(
        job_id="ingest_test",
        workspace_id="workspace",
        project="alpha",
        status="succeeded",
        stage="completed",
        current=1,
        total=1,
        documents_total=1,
        documents_succeeded=1,
        documents_failed=0,
        chunks_total=1,
        current_file=None,
        files_added=1,
        files_updated=0,
        files_skipped=0,
        started_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:01+00:00",
        finished_at="2026-01-01T00:00:01+00:00",
    )
    output_dir = tmp_path / "output"
    atomic_write_json(output_dir / "alpha" / "progress.json", job.to_dict())
    context = FakeContext()

    payload = asyncio.run(brain_status_realtime(str(output_dir), "alpha", cast(Context, context)))

    assert payload["ok"] is True
    assert payload["final"]["status"] == "succeeded"
    assert [event["status"] for event in payload["events"]] == ["succeeded"]
    assert len(context.messages) == len(context.progress) == 1
