from src.config import Config
from src.pipeline import run_pipeline


class FakeEmbeddingClient:
    embedded_texts = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.__class__.embedded_texts = list(texts)
        return [[0.1, 0.2] for _ in texts]


class FakeES:
    indexed_chunks = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def index_docs(self, chunks, embeddings):
        self.__class__.indexed_chunks = list(chunks)
        assert len(embeddings) == len(chunks)
        return "docs_test_current"


def test_run_pipeline_offline_flow(monkeypatch, tmp_path):
    input_dir = tmp_path / "docs"
    input_dir.mkdir()
    (input_dir / "product.txt").write_text("产品资料正文\n\n使用方法：\n\n早晚使用。", encoding="utf-8")

    output_dir = tmp_path / "out"
    monkeypatch.setattr("src.pipeline.EmbeddingClient", FakeEmbeddingClient)
    monkeypatch.setattr("src.pipeline.ESStore", FakeES)

    cfg = Config(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        project="测试项目",
        embedding_url="http://fake-embedding",
        embedding_api_key="fake-key",
        embedding_model="fake-embedding",
        embedding_dim=2,
        es_url="http://fake-es",
        chunk_size=80,
        chunk_overlap=10,
    )

    run_pipeline(cfg)

    assert FakeES.indexed_chunks
    assert {chunk.file_name for chunk in FakeES.indexed_chunks} == {"product.txt"}
    assert any("文档标题：产品资料正文" in text for text in FakeEmbeddingClient.embedded_texts)
    assert any("章节标题：使用方法：" in text for text in FakeEmbeddingClient.embedded_texts)
    assert not (output_dir / "qa_list.json").exists()
    assert not (output_dir / "product_description.md").exists()
