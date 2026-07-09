from __future__ import annotations

import asyncio
import hashlib
import json
from contextlib import asynccontextmanager

from src.models import QAPair, RetrievedChunk, TextChunk


DEFAULT_RRF_K = 60


@asynccontextmanager
async def _es_ctx(
    *,
    url: str = "",
    cloud_id: str = "",
    api_key: str = "",
    username: str = "",
    password: str = "",
):
    from elasticsearch import AsyncElasticsearch

    kwargs: dict = {"verify_certs": True, "request_timeout": 120, "max_retries": 3, "retry_on_timeout": True}
    if cloud_id:
        kwargs["cloud_id"] = cloud_id.strip()
    elif url:
        kwargs["hosts"] = [url.strip().rstrip("/")]
    else:
        raise ValueError("必须提供 es_cloud_id 或 es_url")
    if api_key:
        kwargs["api_key"] = api_key.strip()
    elif username and password:
        kwargs["basic_auth"] = (username.strip(), password.strip())
    client = AsyncElasticsearch(**kwargs)
    try:
        yield client
    finally:
        await client.close()


def _docs_index(workspace_id: str) -> str:
    return f"docs_{workspace_id.lower()}"


def _qa_index(workspace_id: str) -> str:
    return f"qa_{workspace_id.lower()}"


def _run_async(coro):
    """在同步上下文中执行协程。"""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.new_event_loop().run_until_complete(coro)
    raise RuntimeError("不能在已有事件循环中同步调用 ES 方法")


class ESStore:
    """Elasticsearch 索引 + 检索。"""

    def __init__(
        self,
        workspace_id: str,
        *,
        es_url: str = "",
        es_cloud_id: str = "",
        es_user: str = "",
        es_pass: str = "",
        es_api_key: str = "",
        embedding_dim: int = 1024,
    ):
        self.wid = workspace_id
        self.es_url = es_url
        self.es_cloud_id = es_cloud_id
        self.es_user = es_user
        self.es_pass = es_pass
        self.es_api_key = es_api_key
        self.emb_dim = embedding_dim

    def _docs_mapping(self) -> dict:
        return {
            "properties": {
                "id": {"type": "keyword"},
                "workspace_id": {"type": "keyword"},
                "file_name": {"type": "text"},
                "source_path": {"type": "text"},
                "content": {"type": "text", "analyzer": "standard"},
                "page_number": {"type": "integer"},
                "section": {"type": "text"},
                "chunk_type": {"type": "keyword"},
                "metadata": {"type": "object", "enabled": False},
                "embedding": {"type": "dense_vector", "dims": self.emb_dim, "index": True, "similarity": "cosine"},
            }
        }

    def index_docs(self, chunks: list[TextChunk], embeddings: list[list[float]]):
        async def _run():
            async with _es_ctx(
                url=self.es_url,
                cloud_id=self.es_cloud_id,
                api_key=self.es_api_key,
                username=self.es_user,
                password=self.es_pass,
            ) as es:
                idx = _docs_index(self.wid)
                await es.options(ignore_status=404).indices.delete(index=idx)
                await es.indices.create(index=idx, mappings=self._docs_mapping())
                for i in range(0, len(chunks), 50):
                    batch_c = chunks[i : i + 50]
                    batch_e = embeddings[i : i + 50]
                    ops = []
                    for offset, c in enumerate(batch_c):
                        e = batch_e[offset] if offset < len(batch_e) else []
                        ops.append({"index": {"_index": idx, "_id": c.id}})
                        doc = {
                            "id": c.id,
                            "workspace_id": c.workspace_id,
                            "file_name": c.file_name,
                            "source_path": c.source_path,
                            "content": c.content,
                            "page_number": c.page_number,
                            "section": c.section,
                            "chunk_type": c.chunk_type,
                            "metadata": c.metadata,
                        }
                        if e:
                            doc["embedding"] = e
                        ops.append(doc)
                    if ops:
                        await es.bulk(operations=ops, refresh=True)
                print(f"  [索引] docs: {len(chunks)} 条写入完成")

        _run_async(_run())

    def search_docs(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 5,
        vector_k: int = 10,
        keyword_k: int = 20,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> list[RetrievedChunk]:
        """两路独立召回 + RRF 融合，与原始 QASearchService 逻辑一致。"""

        async def _run():
            async with _es_ctx(
                url=self.es_url,
                cloud_id=self.es_cloud_id,
                api_key=self.es_api_key,
                username=self.es_user,
                password=self.es_pass,
            ) as es:
                idx = _docs_index(self.wid)
                if not await es.indices.exists(index=idx):
                    return []

                vector_hits_raw = []
                if query_embedding and vector_k > 0:
                    vector_body: dict = {
                        "size": vector_k,
                        "knn": {
                            "field": "embedding",
                            "query_vector": query_embedding,
                            "k": vector_k,
                            "num_candidates": max(vector_k * 2, 50),
                        },
                    }
                    try:
                        vr = await es.search(index=idx, body=vector_body)
                        vector_hits_raw = vr.get("hits", {}).get("hits", [])
                    except Exception as e:
                        print(f"  [ES] 向量召回失败: {e}")

                keyword_hits_raw = []
                if keyword_k > 0:
                    keyword_body = {
                        "size": keyword_k,
                        "query": {"match": {"content": {"query": query}}},
                    }
                    try:
                        kr = await es.search(index=idx, body=keyword_body)
                        keyword_hits_raw = kr.get("hits", {}).get("hits", [])
                    except Exception as e:
                        print(f"  [ES] 关键词召回失败: {e}")

                def _hit_to_chunk(hit: dict, method: str) -> RetrievedChunk:
                    s = hit.get("_source", {})
                    meta = s.get("metadata") or {}
                    return RetrievedChunk(
                        chunk=TextChunk(
                            id=str(s.get("id", "")),
                            workspace_id=str(s.get("workspace_id", "")),
                            file_name=str(s.get("file_name", "")),
                            source_path=str(s.get("source_path", "")),
                            content=str(s.get("content", "")),
                            page_number=s.get("page_number"),
                            section=str(s.get("section", "")),
                            chunk_type=str(s.get("chunk_type", "paragraph")),
                            metadata={str(k): str(v) for k, v in meta.items()},
                        ),
                        score=float(hit.get("_score", 0)),
                        retrieval_method=method,
                    )

                def _rrf(hits: list, route_name: str) -> dict[str, dict]:
                    scored: dict[str, dict] = {}
                    for rank, h in enumerate(hits, start=1):
                        cid = str(h.get("_source", {}).get("id", ""))
                        if not cid:
                            continue
                        if cid not in scored:
                            scored[cid] = {"item": h, "rrf": 0.0, "raw": 0.0}
                        scored[cid]["rrf"] += 1.0 / (rrf_k + rank)
                        score = float(h.get("_score", 0))
                        if score > scored[cid]["raw"]:
                            scored[cid]["raw"] = score
                            scored[cid]["item"] = h
                    return scored

                vec_map = _rrf(vector_hits_raw[:vector_k], "vector")
                kw_map = _rrf(keyword_hits_raw[:keyword_k], "keyword")

                all_keys = set(vec_map.keys()) | set(kw_map.keys())
                merged = []
                for cid in all_keys:
                    vs = vec_map.get(cid, {})
                    ks = kw_map.get(cid, {})
                    rrf_score = vs.get("rrf", 0.0) + ks.get("rrf", 0.0)
                    best_hit = vs.get("item") or ks.get("item")
                    if best_hit is None:
                        continue
                    route = "rrf"
                    if cid in vec_map and cid not in kw_map:
                        route = "elasticsearch_vector"
                    elif cid in kw_map and cid not in vec_map:
                        route = "elasticsearch_keyword"
                    merged.append((rrf_score, best_hit, route))

                merged.sort(key=lambda x: x[0], reverse=True)
                return [_hit_to_chunk(h, route) for _, h, route in merged[:top_k]]

        return _run_async(_run())

    def _qa_mapping(self) -> dict:
        return {
            "properties": {
                "id": {"type": "keyword"},
                "workspace_id": {"type": "keyword"},
                "question": {"type": "text", "analyzer": "standard"},
                "answer": {"type": "text", "analyzer": "standard"},
                "category": {"type": "keyword"},
                "risk_notes": {"type": "text"},
                "evidence": {"type": "object", "enabled": False},
                "metadata": {"type": "object", "enabled": False},
                "qa_embedding": {"type": "dense_vector", "dims": self.emb_dim, "index": True, "similarity": "cosine"},
            }
        }

    def index_qa(self, qa_pairs: list[QAPair], embeddings: list[list[float]]):
        async def _run():
            async with _es_ctx(
                url=self.es_url,
                cloud_id=self.es_cloud_id,
                api_key=self.es_api_key,
                username=self.es_user,
                password=self.es_pass,
            ) as es:
                idx = _qa_index(self.wid)
                await es.options(ignore_status=404).indices.delete(index=idx)
                await es.indices.create(index=idx, mappings=self._qa_mapping())
                ops = []
                for offset, qa in enumerate(qa_pairs):
                    emb = embeddings[offset] if offset < len(embeddings) else []
                    doc_id = f"qa_{hashlib.md5(qa.question.encode()).hexdigest()[:16]}"
                    doc = {
                        "id": doc_id,
                        "workspace_id": self.wid,
                        "question": qa.question,
                        "answer": qa.answer,
                        "category": qa.category,
                        "risk_notes": " | ".join(qa.risk_notes),
                        "evidence": json.dumps([e.to_dict() for e in qa.evidence], ensure_ascii=False),
                    }
                    if emb:
                        doc["qa_embedding"] = emb
                    ops.append({"index": {"_index": idx, "_id": doc_id}})
                    ops.append(doc)
                if ops:
                    await es.bulk(operations=ops, refresh=True)
                print(f"  [索引] qa: {len(qa_pairs)} 条写入完成")

        _run_async(_run())
