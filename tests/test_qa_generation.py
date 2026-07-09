from src.models import RetrievedChunk, TextChunk
from src.qa.generation import generate_qa_pairs


class FakeLLM:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return []

    def generate_answers_batch(
        self,
        *,
        product_name: str,
        questions: list[tuple[str, str]],
        retrieved_list: list[list[RetrievedChunk]],
    ) -> list[tuple[str, list[str]]]:
        return [(f"{product_name}: {question}", []) for _, question in questions]


class FakeES:
    def search_docs(self, query: str, query_embedding: list[float], top_k: int = 5) -> list[RetrievedChunk]:
        chunk = TextChunk(
            id="chunk_1",
            workspace_id="wid",
            file_name="manual.md",
            source_path="manual.md",
            content=f"{query} 的资料",
            page_number=1,
            section="说明",
            chunk_type="paragraph",
        )
        return [RetrievedChunk(chunk=chunk, score=0.8, retrieval_method="fake")]


def test_generate_qa_pairs_guards_zero_batch_size():
    qa_pairs = generate_qa_pairs(
        product_name="产品A",
        llm=FakeLLM(),
        es=FakeES(),
        questions=[("使用", "怎么用？"), ("功效", "有什么效果？")],
        batch_size=0,
    )

    assert [p.question for p in qa_pairs] == ["怎么用？", "有什么效果？"]
    assert qa_pairs[0].answer == "产品A: 怎么用？"
    assert qa_pairs[0].evidence[0].file_name == "manual.md"
