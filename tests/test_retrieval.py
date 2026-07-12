import pytest

from brain.models import RetrievalOutcome, RetrievedChunk, TextChunk
from brain.retrieval import SearchService


class FakeEmbeddingClient:
    def embed(self, texts):
        assert texts == ["怎么使用？"]
        return [[0.1, 0.2]]


class FakeStore:
    def search_docs(self, query, query_embedding, top_k, vector_k, keyword_k):
        assert query == "怎么使用？"
        assert query_embedding == [0.1, 0.2]
        assert top_k == 3
        assert vector_k == 20
        assert keyword_k == 20
        chunk = TextChunk(
            id="chunk_1",
            workspace_id="wid",
            file_name="manual.md",
            source_path="manual.md",
            content="早晚使用。",
            page_number=2,
            section="使用方法",
            chunk_type="paragraph",
        )
        return RetrievalOutcome(
            results=[RetrievedChunk(chunk=chunk, score=0.03, retrieval_method="rrf")],
            warnings=[],
        )


def test_search_service_embeds_and_retrieves_query():
    outcome = SearchService(FakeEmbeddingClient(), FakeStore()).search(" 怎么使用？ ", top_k=3)

    assert outcome.results[0].chunk.content == "早晚使用。"


@pytest.mark.parametrize("query, top_k", [("", 5), ("问题", 0), ("问题", 101)])
def test_search_service_validates_input(query, top_k):
    with pytest.raises(ValueError):
        SearchService(FakeEmbeddingClient(), FakeStore()).search(query, top_k=top_k)
