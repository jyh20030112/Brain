import json

from src.config import Config
from src.models import RetrievedChunk
from src.pipeline import run_pipeline


class FakeLLM:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def generate_product_description(self, context: str) -> str:
        assert "产品资料正文" in context
        return "# 产品A\n\n产品描述。"

    def generate_customer_questions(self, product_description: str, count: int, generalization_count: int) -> list[dict]:
        assert product_description.startswith("# 产品A")
        return [{"category": "使用方法", "question": "怎么用？", "variations": ["使用步骤是什么？"]}]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2] for _ in texts]

    def generate_answers_batch(
        self,
        *,
        product_name: str,
        questions: list[tuple[str, str]],
        retrieved_list: list[list[RetrievedChunk]],
    ) -> list[tuple[str, list[str]]]:
        return [(f"{product_name} 可以按资料说明使用。", []) for _ in questions]


class FakeES:
    indexed_chunks = []
    indexed_qa = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def index_docs(self, chunks, embeddings):
        self.__class__.indexed_chunks = list(chunks)
        assert len(embeddings) == len(chunks)

    def search_docs(self, query: str, query_embedding: list[float], top_k: int = 5):
        return [RetrievedChunk(chunk=self.indexed_chunks[0], score=0.9, retrieval_method="fake")]

    def index_qa(self, qa_pairs, embeddings):
        self.__class__.indexed_qa = list(qa_pairs)
        assert len(embeddings) == len(qa_pairs)


def test_run_pipeline_offline_flow(monkeypatch, tmp_path):
    input_dir = tmp_path / "docs"
    input_dir.mkdir()
    (input_dir / "product.txt").write_text("产品资料正文\n\n使用方法：\n\n早晚使用。", encoding="utf-8")

    output_dir = tmp_path / "out"
    monkeypatch.setattr("src.pipeline.LLMClient", FakeLLM)
    monkeypatch.setattr("src.pipeline.ESStore", FakeES)

    cfg = Config(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        project="测试项目",
        llm_base_url="http://fake-llm",
        llm_api_key="fake-key",
        llm_model="fake-model",
        embedding_model="fake-embedding",
        embedding_dim=2,
        es_url="http://fake-es",
        qa_limit=1,
        qa_generalization=1,
        chunk_size=80,
        chunk_overlap=10,
    )

    run_pipeline(cfg)

    qa_json = json.loads((output_dir / "qa_list.json").read_text(encoding="utf-8"))
    assert qa_json["product_name"] == "产品A"
    assert len(qa_json["qa_pairs"]) == 2
    assert (output_dir / "qa_list.md").exists()
    assert (output_dir / "qa_list.csv").exists()
    assert (output_dir / "product_description.md").read_text(encoding="utf-8").startswith("# 产品A")
    assert FakeES.indexed_chunks
    assert len(FakeES.indexed_qa) == 2
