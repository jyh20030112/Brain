from types import SimpleNamespace

import pytest

from brain.config import Config
from brain.ingestion import run_ingestion


class FakeEmbeddingClient:
    embedded_texts = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def embed(self, texts: list[str], progress_callback=None) -> list[list[float]]:
        self.__class__.embedded_texts = list(texts)
        if progress_callback:
            progress_callback(len(texts), len(texts))
        return [[0.1, 0.2] for _ in texts]


class FakeES:
    indexed_chunks = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def index_docs(self, chunks, embeddings, *, progress_callback=None, publishing_callback=None):
        self.__class__.indexed_chunks = list(chunks)
        assert len(embeddings) == len(chunks)
        if progress_callback:
            progress_callback(len(chunks), len(chunks))
        if publishing_callback:
            publishing_callback()
        return "docs_test_current"


class FakeProgressStore:
    def __init__(self):
        self.updates = []
        self.completed = None
        self.failed = None

    def create_job(self, *, project, workspace_id):
        return SimpleNamespace(job_id="ingest_test")

    def update_job(self, job_id, **fields):
        self.updates.append(fields)

    def complete_job(self, job_id, *, active_index):
        self.completed = active_index

    def fail_job(self, job_id, *, error):
        self.failed = error


def test_run_ingestion_offline_flow(monkeypatch, tmp_path):
    input_dir = tmp_path / "docs"
    input_dir.mkdir()
    (input_dir / "manual.txt").write_text("知识库资料正文\n\n使用方法：\n\n按照说明操作。", encoding="utf-8")

    output_dir = tmp_path / "out"
    progress = FakeProgressStore()
    monkeypatch.setattr("brain.ingestion.build_embedding_client", lambda cfg: FakeEmbeddingClient())
    monkeypatch.setattr("brain.ingestion.build_es_store", lambda cfg: FakeES())
    monkeypatch.setattr("brain.ingestion.build_progress_store", lambda cfg: progress)

    cfg = Config(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        project="test-knowledge-base",
        embedding_url="http://fake-embedding",
        embedding_api_key="fake-key",
        embedding_model="fake-embedding",
        embedding_dim=2,
        es_url="http://fake-es",
        chunk_size=80,
        chunk_overlap=10,
    )

    assert run_ingestion(cfg) == "ingest_test"

    assert FakeES.indexed_chunks
    assert {chunk.file_name for chunk in FakeES.indexed_chunks} == {"manual.txt"}
    assert any("文档标题：知识库资料正文" in text for text in FakeEmbeddingClient.embedded_texts)
    assert any("章节标题：使用方法：" in text for text in FakeEmbeddingClient.embedded_texts)
    assert {update["stage"] for update in progress.updates} >= {
        "scanning",
        "parsing",
        "cleaning",
        "chunking",
        "embedding",
        "indexing",
        "publishing",
    }
    assert progress.completed == "docs_test_current"
    assert progress.failed is None
    assert not (output_dir / "qa_list.json").exists()
    assert not (output_dir / "product_description.md").exists()


def test_run_ingestion_records_failed_status(monkeypatch, tmp_path):
    input_dir = tmp_path / "docs"
    input_dir.mkdir()
    (input_dir / "manual.txt").write_text("资料正文。", encoding="utf-8")
    progress = FakeProgressStore()

    class FailingEmbeddingClient:
        def embed(self, texts, progress_callback=None):
            raise RuntimeError("embedding unavailable")

    monkeypatch.setattr("brain.ingestion.build_embedding_client", lambda cfg: FailingEmbeddingClient())
    monkeypatch.setattr("brain.ingestion.build_es_store", lambda cfg: FakeES())
    monkeypatch.setattr("brain.ingestion.build_progress_store", lambda cfg: progress)
    cfg = Config(
        input_dir=str(input_dir),
        output_dir=str(tmp_path / "out"),
        project="test-knowledge-base",
        embedding_url="http://fake-embedding",
        embedding_model="fake-embedding",
        embedding_dim=2,
        es_url="http://fake-es",
        chunk_size=80,
        chunk_overlap=10,
    )

    with pytest.raises(RuntimeError, match="embedding unavailable"):
        run_ingestion(cfg)

    assert progress.completed is None
    assert "embedding unavailable" in progress.failed
