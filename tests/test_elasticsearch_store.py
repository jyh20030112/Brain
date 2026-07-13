import math
from contextlib import asynccontextmanager

import pytest

from simbrain.storage.elasticsearch_store import (
    ESStore,
    ProjectNotFoundError,
    RetrievalFailedError,
    _docs_index,
)
from simbrain.models import TextChunk


def _chunk() -> TextChunk:
    return TextChunk(
        id="chunk_1",
        workspace_id="wid",
        file_name="source.md",
        source_path="source.md",
        content="原始资料",
        page_number=1,
        section="说明",
        chunk_type="paragraph",
    )


def test_docs_index_is_a_current_alias():
    assert _docs_index("ABCD") == "docs_abcd_current"


@pytest.mark.parametrize(
    "embeddings, message",
    [
        ([], "数量不匹配"),
        ([[0.1]], "维度"),
        ([[0.1, math.nan]], "非有限"),
    ],
)
def test_validate_embeddings_rejects_incomplete_or_invalid_vectors(embeddings, message):
    store = ESStore("wid", embedding_dim=2)

    with pytest.raises(ValueError, match=message):
        store._validate_embeddings([_chunk()], embeddings)


def test_validate_embeddings_accepts_complete_vectors():
    ESStore("wid", embedding_dim=2)._validate_embeddings([_chunk()], [[0.1, 0.2]])


async def fake_streaming_bulk(es, actions, **kwargs):
    for action in actions:
        es.operations.append(action)
        if es.bulk_response.get("errors"):
            yield False, es.bulk_response["items"][0]
        else:
            es.document_count += 1
            yield True, {"index": {"_id": action["_id"]}}


class FakeIndices:
    def __init__(self, client, aliases):
        self.client = client
        self.aliases = aliases

    async def create(self, *, index, mappings):
        self.client.created.append(index)
        self.client.versions[index] = {
            "settings": {"index": {"creation_date": str(len(self.client.versions) + 2)}}
        }

    async def exists(self, *, index):
        return bool(self.aliases)

    async def refresh(self, *, index):
        self.client.refreshed.append(index)

    async def exists_alias(self, *, name):
        return bool(self.aliases)

    async def get_alias(self, *, name):
        return self.aliases

    async def update_aliases(self, *, actions):
        self.client.alias_actions = actions

    async def get(self, **kwargs):
        return self.client.versions

    async def delete(self, *, index):
        names = index if isinstance(index, list) else [index]
        self.client.deleted.extend(names)


class FakeESClient:
    def __init__(self, *, aliases, bulk_response, old_count=1):
        self.created = []
        self.refreshed = []
        self.deleted = []
        self.alias_actions = []
        self.indices = FakeIndices(self, aliases)
        self.bulk_response = bulk_response
        self.old_count = old_count
        self.document_count = 0
        self.delete_queries = []
        self.operations = []
        self.versions = {
            name: {"settings": {"index": {"creation_date": "1"}}}
            for name in aliases
        }

    def options(self, **kwargs):
        return self

    async def count(self, *, index):
        return {"count": self.document_count}

    async def reindex(self, **kwargs):
        self.document_count = self.old_count
        return {"failures": []}

    async def delete_by_query(self, **kwargs):
        self.delete_queries.append(kwargs["query"])
        deleted = self.document_count
        self.document_count = 0
        return {"deleted": deleted, "failures": []}


def test_incremental_publish_replaces_matching_file_and_switches_alias(monkeypatch):
    alias = _docs_index("wid")
    client = FakeESClient(aliases={"docs_wid_current_v_old": {"aliases": {alias: {}}}}, bulk_response={"errors": False})

    @asynccontextmanager
    async def fake_context(**kwargs):
        yield client

    monkeypatch.setattr("simbrain.storage.elasticsearch_store.es_context", fake_context)
    monkeypatch.setattr("simbrain.storage.elasticsearch_store.async_streaming_bulk", fake_streaming_bulk)
    store = ESStore("wid", embedding_dim=2)

    async def fake_inventory(es, index):
        return [{"file_name": "source.md", "chunk_count": 1}]

    monkeypatch.setattr(store, "_inventory", fake_inventory)
    progress = []
    publishing = []
    prepared = []
    published = store.publish_incremental(
        [_chunk()],
        [[0.1, 0.2]],
        replace_file_names=["source.md"],
        progress_callback=lambda current, total: progress.append((current, total)),
        publishing_callback=lambda: publishing.append(True),
        prepare_manifest_callback=lambda inventory, alias, version: prepared.append((inventory, alias, version)),
    )

    assert published.alias == alias
    staging = client.created[0]
    assert staging.startswith(f"{alias}_v_")
    assert client.refreshed == [staging]
    assert client.alias_actions == [
        {"remove": {"index": "docs_wid_current_v_old", "alias": alias}},
        {"add": {"index": staging, "alias": alias}},
    ]
    assert client.deleted == []
    assert progress == [(1, 1)]
    assert publishing == [True]
    assert prepared[0][0][0]["file_name"] == "source.md"
    should = client.delete_queries[0]["bool"]["should"]
    assert should[0]["terms"]["file_name_normalized"] == ["source.md"]
    assert should[1]["wildcard"]["file_name"]["case_insensitive"] is True


def test_incremental_publish_keeps_current_alias_when_bulk_write_fails(monkeypatch):
    client = FakeESClient(
        aliases={"docs_wid_current_v_old": {"aliases": {_docs_index("wid"): {}}}},
        bulk_response={"errors": True, "items": [{"index": {"_id": "chunk_1", "error": "bad vector"}}]},
    )

    @asynccontextmanager
    async def fake_context(**kwargs):
        yield client

    monkeypatch.setattr("simbrain.storage.elasticsearch_store.es_context", fake_context)
    monkeypatch.setattr("simbrain.storage.elasticsearch_store.async_streaming_bulk", fake_streaming_bulk)

    with pytest.raises(RuntimeError, match="批量入库失败"):
        ESStore("wid", embedding_dim=2).publish_incremental(
            [_chunk()],
            [[0.1, 0.2]],
            replace_file_names=["source.md"],
        )

    assert client.alias_actions == []
    assert client.deleted == client.created


def test_incremental_publish_removes_obsolete_index_versions(monkeypatch):
    alias = _docs_index("wid")
    client = FakeESClient(
        aliases={"docs_wid_current_v_old": {"aliases": {alias: {}}}},
        bulk_response={"errors": False},
    )

    @asynccontextmanager
    async def fake_context(**kwargs):
        yield client

    monkeypatch.setattr("simbrain.storage.elasticsearch_store.es_context", fake_context)
    monkeypatch.setattr("simbrain.storage.elasticsearch_store.async_streaming_bulk", fake_streaming_bulk)
    store = ESStore("wid", embedding_dim=2, index_versions_to_keep=1)
    monkeypatch.setattr(store, "_inventory", lambda es, index: _async_value([{"file_name": "source.md", "chunk_count": 1}]))

    store.publish_incremental([_chunk()], [[0.1, 0.2]], replace_file_names=["source.md"])

    assert "docs_wid_current_v_old" in client.deleted


async def _async_value(value):
    return value


class SearchIndices:
    def __init__(self, exists=True):
        self.exists = exists

    async def exists_alias(self, *, name):
        return self.exists


class SearchClient:
    def __init__(self, *, vector=None, keyword=None, exists=True):
        self.indices = SearchIndices(exists)
        self.vector = vector
        self.keyword = keyword

    async def search(self, *, index, body):
        response = self.vector if "knn" in body else self.keyword
        if isinstance(response, Exception):
            raise response
        return {"hits": {"hits": response or []}}


def _hit(chunk_id, score=1.0):
    return {
        "_score": score,
        "_source": {
            "id": chunk_id,
            "workspace_id": "wid",
            "file_name": f"{chunk_id}.md",
            "source_path": f"{chunk_id}.md",
            "content": chunk_id,
            "page_number": 1,
            "section": "说明",
            "chunk_type": "paragraph",
            "metadata": {},
        },
    }


def _install_search_client(monkeypatch, client):
    @asynccontextmanager
    async def fake_context(**kwargs):
        yield client

    monkeypatch.setattr("simbrain.storage.elasticsearch_store.es_context", fake_context)


def test_search_rejects_missing_project(monkeypatch):
    _install_search_client(monkeypatch, SearchClient(exists=False))

    with pytest.raises(ProjectNotFoundError):
        ESStore("wid", embedding_dim=2).search_docs("问题", [0.1, 0.2])


def test_search_returns_warning_when_one_route_degrades(monkeypatch):
    _install_search_client(
        monkeypatch,
        SearchClient(vector=RuntimeError("vector unavailable"), keyword=[_hit("chunk_1")]),
    )

    outcome = ESStore("wid", embedding_dim=2).search_docs("问题", [0.1, 0.2])

    assert [item.chunk.id for item in outcome.results] == ["chunk_1"]
    assert outcome.warnings[0]["code"] == "vector_retrieval_failed"


def test_search_fails_when_all_routes_fail(monkeypatch):
    _install_search_client(
        monkeypatch,
        SearchClient(vector=RuntimeError("vector unavailable"), keyword=RuntimeError("keyword unavailable")),
    )

    with pytest.raises(RetrievalFailedError, match="所有召回路线均失败"):
        ESStore("wid", embedding_dim=2).search_docs("问题", [0.1, 0.2])


def test_rrf_ties_are_sorted_by_chunk_id(monkeypatch):
    _install_search_client(
        monkeypatch,
        SearchClient(vector=[_hit("b")], keyword=[_hit("a")]),
    )

    outcome = ESStore("wid", embedding_dim=2).search_docs("问题", [0.1, 0.2], top_k=2)

    assert [item.chunk.id for item in outcome.results] == ["a", "b"]
