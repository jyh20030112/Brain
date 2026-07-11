from __future__ import annotations

import asyncio
import math
import sys
from contextlib import asynccontextmanager
from numbers import Real
from uuid import uuid4

from brain.models import RetrievedChunk, TextChunk


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
    """当前可查询的 docs 索引别名。"""
    return f"docs_{workspace_id.lower()}_current"


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

    def _validate_embeddings(self, chunks: list[TextChunk], embeddings: list[list[float]]) -> None:
        if len(embeddings) != len(chunks):
            raise ValueError(f"向量数量不匹配：获得 {len(embeddings)}，预期 {len(chunks)}")
        for position, vector in enumerate(embeddings, start=1):
            if not isinstance(vector, (list, tuple)):
                raise ValueError(f"第 {position} 条向量不是数组")
            if len(vector) != self.emb_dim:
                raise ValueError(f"第 {position} 条向量维度为 {len(vector)}，预期 {self.emb_dim}")
            if not all(isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value) for value in vector):
                raise ValueError(f"第 {position} 条向量包含非有限数值")

    @staticmethod
    def _raise_on_bulk_errors(response: dict) -> None:
        body = getattr(response, "body", response)
        if not body.get("errors"):
            return
        errors = []
        for item in body.get("items", []):
            operation = item.get("index", {})
            if "error" in operation:
                errors.append(f"{operation.get('_id', '<unknown>')}: {operation['error']}")
            if len(errors) == 3:
                break
        detail = "；".join(errors) or "Elasticsearch bulk 返回 errors=true"
        raise RuntimeError(f"批量入库失败：{detail}")

    def index_docs(self, chunks: list[TextChunk], embeddings: list[list[float]]) -> str:
        """写入新版本索引；全部成功并校验数量后，原子切换查询别名。"""
        self._validate_embeddings(chunks, embeddings)

        async def _run():
            async with _es_ctx(
                url=self.es_url,
                cloud_id=self.es_cloud_id,
                api_key=self.es_api_key,
                username=self.es_user,
                password=self.es_pass,
            ) as es:
                alias = _docs_index(self.wid)
                staging_index = f"{alias}_v_{uuid4().hex[:12]}"
                await es.indices.create(index=staging_index, mappings=self._docs_mapping())
                try:
                    for i in range(0, len(chunks), 50):
                        batch_c = chunks[i : i + 50]
                        batch_e = embeddings[i : i + 50]
                        ops = []
                        for c, vector in zip(batch_c, batch_e, strict=True):
                            ops.append({"index": {"_index": staging_index, "_id": c.id}})
                            ops.append(
                                {
                                    "id": c.id,
                                    "workspace_id": c.workspace_id,
                                    "file_name": c.file_name,
                                    "source_path": c.source_path,
                                    "content": c.content,
                                    "page_number": c.page_number,
                                    "section": c.section,
                                    "chunk_type": c.chunk_type,
                                    "metadata": c.metadata,
                                    "embedding": vector,
                                }
                            )
                        if ops:
                            response = await es.bulk(operations=ops, refresh=False)
                            self._raise_on_bulk_errors(response)

                    await es.indices.refresh(index=staging_index)
                    count_response = await es.count(index=staging_index)
                    indexed = getattr(count_response, "body", count_response)
                    if int(indexed.get("count", -1)) != len(chunks):
                        raise RuntimeError(f"索引计数校验失败：实际 {indexed.get('count')}，预期 {len(chunks)}")

                    existing = await es.options(ignore_status=404).indices.get_alias(name=alias)
                    existing_body = getattr(existing, "body", existing)
                    previous_indices = sorted(existing_body.keys()) if hasattr(existing_body, "keys") else []
                    actions = [{"remove": {"index": index, "alias": alias}} for index in previous_indices]
                    actions.append({"add": {"index": staging_index, "alias": alias}})
                    await es.indices.update_aliases(actions=actions)
                except Exception:
                    await es.options(ignore_status=404).indices.delete(index=staging_index)
                    raise

                print(f"  [索引] docs: {len(chunks)} 条写入完成，已发布到 {alias}")
                return alias

        return _run_async(_run())

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
                        print(f"  [ES] 向量召回失败: {e}", file=sys.stderr)

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
                        print(f"  [ES] 关键词召回失败: {e}", file=sys.stderr)

                def _hit_to_chunk(hit: dict, method: str, score: float) -> RetrievedChunk:
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
                        score=score,
                        retrieval_method=method,
                    )

                def _rrf(hits: list) -> dict[str, dict]:
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

                vec_map = _rrf(vector_hits_raw[:vector_k])
                kw_map = _rrf(keyword_hits_raw[:keyword_k])

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
                return [_hit_to_chunk(h, route, score) for score, h, route in merged[:top_k]]

        return _run_async(_run())
