from __future__ import annotations

from brain.config import Config
from brain.embeddings import EmbeddingClient
from brain.models import RetrievedChunk
from brain.runtime import build_embedding_client, build_es_store
from brain.storage.elasticsearch_store import ESStore


class SearchService:
    """独立检索服务：生成查询向量并召回知识块。"""

    def __init__(self, embedding_client: EmbeddingClient, store: ESStore):
        self.embedding_client = embedding_client
        self.store = store

    @classmethod
    def from_config(cls, cfg: Config) -> SearchService:
        cfg.validate_for_search()
        return cls(build_embedding_client(cfg), build_es_store(cfg))

    def search(self, query: str, *, top_k: int = 5) -> list[RetrievedChunk]:
        query = query.strip()
        if not query:
            raise ValueError("查询内容不能为空")
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        query_embedding = self.embedding_client.embed([query])[0]
        return self.store.search_docs(query, query_embedding, top_k=top_k)
