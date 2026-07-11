import json

from brain.cli import ingest, search
from brain.models import RetrievedChunk, TextChunk


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

    assert ingest.main(["--input-dir", "docs", "--project", "产品库"]) == 0
    assert captured["cfg"].input_dir == "docs"
    assert captured["cfg"].project == "产品库"
