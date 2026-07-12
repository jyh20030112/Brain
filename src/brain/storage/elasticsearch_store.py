from __future__ import annotations

import asyncio
import math
import sys
from dataclasses import dataclass
from numbers import Real
from typing import Any, Callable
from uuid import uuid4

from elasticsearch.helpers import async_streaming_bulk

from brain.models import RetrievalOutcome, RetrievedChunk, TextChunk
from brain.storage.client import es_context, response_body, run_async


DEFAULT_RRF_K = 60


@dataclass(slots=True)
class PublishResult:
    alias: str
    total_chunks: int


class ProjectNotFoundError(RuntimeError):
    pass


class RetrievalFailedError(RuntimeError):
    pass


def _escape_wildcard(value: str) -> str:
    return value.replace("\\", "\\\\").replace("*", "\\*").replace("?", "\\?")


def _docs_index(workspace_id: str) -> str:
    """当前可查询的 docs 索引别名。"""
    return f"docs_{workspace_id.lower()}_current"


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
        index_versions_to_keep: int = 2,
    ):
        self.wid = workspace_id
        self.es_url = es_url
        self.es_cloud_id = es_cloud_id
        self.es_user = es_user
        self.es_pass = es_pass
        self.es_api_key = es_api_key
        self.emb_dim = embedding_dim
        self.index_versions_to_keep = max(1, index_versions_to_keep)

    def _docs_mapping(self) -> dict:
        return {
            "properties": {
                "id": {"type": "keyword"},
                "workspace_id": {"type": "keyword"},
                "file_name": {"type": "keyword"},
                "file_name_normalized": {"type": "keyword"},
                "source_path": {"type": "keyword"},
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

    async def _inventory(self, es, index: str) -> list[dict[str, Any]]:
        inventory: list[dict[str, Any]] = []
        after: dict | None = None
        while True:
            composite: dict[str, Any] = {
                "size": 500,
                "sources": [{"file_name": {"terms": {"field": "file_name"}}}],
            }
            if after:
                composite["after"] = after
            response = response_body(
                await es.search(
                    index=index,
                    body={
                        "size": 0,
                        "aggs": {
                            "files": {
                                "composite": composite,
                                "aggs": {
                                    "sample": {
                                        "top_hits": {
                                            "size": 1,
                                            "_source": {
                                                "includes": ["file_name", "source_path", "page_number", "metadata"]
                                            },
                                        }
                                    },
                                    "last_page": {"max": {"field": "page_number"}},
                                },
                            }
                        },
                    },
                )
            )
            files = response.get("aggregations", {}).get("files", {})
            buckets = files.get("buckets", [])
            for bucket in buckets:
                hits = bucket.get("sample", {}).get("hits", {}).get("hits", [])
                source = hits[0].get("_source", {}) if hits else {}
                metadata = source.get("metadata") or {}
                file_name = str(source.get("file_name") or bucket.get("key", {}).get("file_name", ""))
                page_count = metadata.get("document_page_count")
                if page_count is None:
                    page_count = bucket.get("last_page", {}).get("value") or 0
                inventory.append(
                    {
                        "file_name": file_name,
                        "source_path": str(source.get("source_path") or file_name),
                        "file_type": str(metadata.get("extension") or "unknown"),
                        "title": str(metadata.get("document_title") or ""),
                        "parser": str(metadata.get("parser") or "legacy"),
                        "mineru_artifact": str(metadata.get("mineru_artifact") or "") or None,
                        "page_count": int(page_count),
                        "chunk_count": int(bucket.get("doc_count", 0)),
                    }
                )
            after = files.get("after_key")
            if not buckets or not after:
                break
        return inventory

    async def _cleanup_old_versions(self, es, alias: str, active_index: str) -> None:
        response = response_body(
            await es.indices.get(
                index=f"{alias}_v_*",
                expand_wildcards="all",
                allow_no_indices=True,
                ignore_unavailable=True,
            )
        )
        versions = []
        for name, details in response.items():
            creation_date = details.get("settings", {}).get("index", {}).get("creation_date", "0")
            versions.append((int(creation_date), name))
        versions.sort(reverse=True)
        keep = {active_index}
        other_versions = [name for _, name in versions if name != active_index]
        keep.update(other_versions[: self.index_versions_to_keep - 1])
        obsolete = [name for _, name in versions if name not in keep]
        if obsolete:
            await es.indices.delete(index=obsolete)

    def publish_incremental(
        self,
        chunks: list[TextChunk],
        embeddings: list[list[float]],
        *,
        replace_file_names: list[str],
        progress_callback: Callable[[int, int], None] | None = None,
        publishing_callback: Callable[[], None] | None = None,
        prepare_manifest_callback: Callable[[list[dict[str, Any]], str, str], None] | None = None,
        abort_manifest_callback: Callable[[], None] | None = None,
    ) -> PublishResult:
        """复制当前索引、替换同名文件并原子发布新版本。"""
        self._validate_embeddings(chunks, embeddings)

        async def _run():
            async with es_context(
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
                    alias_exists = bool(await es.indices.exists_alias(name=alias))
                    if alias_exists:
                        reindex_response = response_body(
                            await es.reindex(
                                source={"index": alias},
                                dest={"index": staging_index},
                                wait_for_completion=True,
                                refresh=True,
                            )
                        )
                        if reindex_response.get("failures"):
                            raise RuntimeError(f"复制当前索引失败: {reindex_response['failures'][:3]}")

                    for start in range(0, len(replace_file_names), 500):
                        names = replace_file_names[start : start + 500]
                        delete_response = response_body(
                            await es.delete_by_query(
                                index=staging_index,
                                query={
                                    "bool": {
                                        "should": [
                                            {"terms": {"file_name_normalized": [name.casefold() for name in names]}},
                                            *[
                                                {
                                                    "wildcard": {
                                                        "file_name": {
                                                            "value": _escape_wildcard(name),
                                                            "case_insensitive": True,
                                                        }
                                                    }
                                                }
                                                for name in names
                                            ],
                                        ],
                                        "minimum_should_match": 1,
                                    }
                                },
                                conflicts="proceed",
                                refresh=True,
                            )
                        )
                        if delete_response.get("failures"):
                            raise RuntimeError(f"删除旧文件 chunks 失败: {delete_response['failures'][:3]}")

                    retained_response = response_body(await es.count(index=staging_index))
                    retained_count = int(retained_response.get("count", 0))
                    actions = (
                        {
                            "_op_type": "index",
                            "_index": staging_index,
                            "_id": c.id,
                            "_source": {
                                "id": c.id,
                                "workspace_id": c.workspace_id,
                                "file_name": c.file_name,
                                "file_name_normalized": c.file_name.casefold(),
                                "source_path": c.source_path,
                                "content": c.content,
                                "page_number": c.page_number,
                                "section": c.section,
                                "chunk_type": c.chunk_type,
                                "metadata": c.metadata,
                                "embedding": vector,
                            },
                        }
                        for c, vector in zip(chunks, embeddings, strict=True)
                    )
                    indexed_count = 0
                    bulk_errors: list[str] = []
                    async for ok, info in async_streaming_bulk(
                        es,
                        actions,
                        chunk_size=50,
                        max_retries=3,
                        initial_backoff=1,
                        max_backoff=8,
                        raise_on_error=False,
                        raise_on_exception=True,
                    ):
                        operation = info.get("index", {})
                        if ok:
                            indexed_count += 1
                            if progress_callback:
                                if indexed_count % 50 == 0 or indexed_count == len(chunks):
                                    await asyncio.to_thread(progress_callback, indexed_count, len(chunks))
                        elif len(bulk_errors) < 3:
                            bulk_errors.append(
                                f"{operation.get('_id', '<unknown>')}: "
                                f"{operation.get('error', 'unknown error')}"
                            )
                    if bulk_errors:
                        raise RuntimeError(f"批量入库失败：{'；'.join(bulk_errors)}")

                    await es.indices.refresh(index=staging_index)
                    count_response = await es.count(index=staging_index)
                    indexed = response_body(count_response)
                    expected_count = retained_count + len(chunks)
                    if int(indexed.get("count", -1)) != expected_count:
                        raise RuntimeError(f"索引计数校验失败：实际 {indexed.get('count')}，预期 {expected_count}")

                    if publishing_callback:
                        await asyncio.to_thread(publishing_callback)
                    inventory = await self._inventory(es, staging_index)
                    if prepare_manifest_callback:
                        await asyncio.to_thread(
                            prepare_manifest_callback,
                            inventory,
                            alias,
                            staging_index,
                        )
                    previous_indices = []
                    if alias_exists:
                        existing = await es.indices.get_alias(name=alias)
                        existing_body = response_body(existing)
                        previous_indices = sorted(existing_body.keys())
                    actions = [{"remove": {"index": index, "alias": alias}} for index in previous_indices]
                    actions.append({"add": {"index": staging_index, "alias": alias}})
                    await es.indices.update_aliases(actions=actions)
                except Exception:
                    await es.options(ignore_status=404).indices.delete(index=staging_index)
                    if abort_manifest_callback:
                        await asyncio.to_thread(abort_manifest_callback)
                    raise

                try:
                    await self._cleanup_old_versions(es, alias, staging_index)
                except Exception as exc:
                    print(f"  [索引] 清理旧版本失败，将在下次入库重试: {exc}", file=sys.stderr)

                print(f"  [索引] docs: 新写入 {len(chunks)} 条，保留 {retained_count} 条，已发布到 {alias}", file=sys.stderr)
                return PublishResult(alias=alias, total_chunks=expected_count)

        return run_async(_run())

    def alias_indices(self) -> list[str]:
        async def _run():
            async with es_context(
                url=self.es_url,
                cloud_id=self.es_cloud_id,
                api_key=self.es_api_key,
                username=self.es_user,
                password=self.es_pass,
            ) as es:
                alias = _docs_index(self.wid)
                if not await es.indices.exists_alias(name=alias):
                    return []
                response = response_body(await es.indices.get_alias(name=alias))
                return sorted(response.keys()) if hasattr(response, "keys") else []

        return run_async(_run())

    def active_index_state(self) -> dict[str, Any]:
        async def _run():
            async with es_context(
                url=self.es_url,
                cloud_id=self.es_cloud_id,
                api_key=self.es_api_key,
                username=self.es_user,
                password=self.es_pass,
            ) as es:
                alias = _docs_index(self.wid)
                if not await es.indices.exists_alias(name=alias):
                    return {"alias": alias, "indices": [], "chunk_count": 0}
                aliases = response_body(await es.indices.get_alias(name=alias))
                count = response_body(await es.count(index=alias))
                return {
                    "alias": alias,
                    "indices": sorted(aliases.keys()),
                    "chunk_count": int(count.get("count", 0)),
                }

        return run_async(_run())

    def search_docs(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 5,
        vector_k: int = 10,
        keyword_k: int = 20,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> RetrievalOutcome:
        """两路独立召回 + RRF 融合，并显式返回检索降级信息。"""

        async def _run():
            async with es_context(
                url=self.es_url,
                cloud_id=self.es_cloud_id,
                api_key=self.es_api_key,
                username=self.es_user,
                password=self.es_pass,
            ) as es:
                idx = _docs_index(self.wid)
                if not await es.indices.exists_alias(name=idx):
                    raise ProjectNotFoundError(f"project 对应的 Elasticsearch 索引不存在: {idx}")

                vector_hits_raw = []
                warnings: list[dict[str, str]] = []
                successful_routes = 0
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
                        vector_hits_raw = response_body(vr).get("hits", {}).get("hits", [])
                        successful_routes += 1
                    except Exception as e:
                        warnings.append({"route": "vector", "code": "vector_retrieval_failed", "message": str(e)})

                keyword_hits_raw = []
                if keyword_k > 0:
                    keyword_body = {
                        "size": keyword_k,
                        "query": {"match": {"content": {"query": query}}},
                    }
                    try:
                        kr = await es.search(index=idx, body=keyword_body)
                        keyword_hits_raw = response_body(kr).get("hits", {}).get("hits", [])
                        successful_routes += 1
                    except Exception as e:
                        warnings.append({"route": "keyword", "code": "keyword_retrieval_failed", "message": str(e)})

                if successful_routes == 0:
                    detail = "；".join(f"{item['route']}: {item['message']}" for item in warnings)
                    raise RetrievalFailedError(f"所有召回路线均失败：{detail or '没有启用召回路线'}")

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

                merged.sort(key=lambda item: (-item[0], str(item[1].get("_source", {}).get("id", ""))))
                results = [_hit_to_chunk(h, route, score) for score, h, route in merged[:top_k]]
                return RetrievalOutcome(results=results, warnings=warnings)

        return run_async(_run())
