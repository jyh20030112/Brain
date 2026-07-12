import math
from contextlib import asynccontextmanager

import pytest

from brain.models import TextChunk
from brain.storage.elasticsearch_store import ESStore, _docs_index


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


def test_bulk_errors_are_raised():
    with pytest.raises(RuntimeError, match="chunk_1"):
        ESStore._raise_on_bulk_errors(
            {"errors": True, "items": [{"index": {"_id": "chunk_1", "error": "bad vector"}}]}
        )


class FakeIndices:
    def __init__(self, client, aliases):
        self.client = client
        self.aliases = aliases

    async def create(self, *, index, mappings):
        self.client.created.append(index)

    async def exists(self, *, index):
        return bool(self.aliases)

    async def refresh(self, *, index):
        self.client.refreshed.append(index)

    async def get_alias(self, *, name):
        return self.aliases

    async def update_aliases(self, *, actions):
        self.client.alias_actions = actions

    async def delete(self, *, index):
        self.client.deleted.append(index)


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

    def options(self, **kwargs):
        return self

    async def bulk(self, *, operations, refresh):
        self.operations = operations
        if not self.bulk_response.get("errors"):
            self.document_count += len(operations) // 2
        return self.bulk_response

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

    monkeypatch.setattr("brain.storage.elasticsearch_store.es_context", fake_context)
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
    assert client.delete_queries[0]["bool"]["should"][0]["wildcard"]["file_name"]["case_insensitive"] is True


def test_incremental_publish_keeps_current_alias_when_bulk_write_fails(monkeypatch):
    client = FakeESClient(
        aliases={"docs_wid_current_v_old": {"aliases": {_docs_index("wid"): {}}}},
        bulk_response={"errors": True, "items": [{"index": {"_id": "chunk_1", "error": "bad vector"}}]},
    )

    @asynccontextmanager
    async def fake_context(**kwargs):
        yield client

    monkeypatch.setattr("brain.storage.elasticsearch_store.es_context", fake_context)

    with pytest.raises(RuntimeError, match="批量入库失败"):
        ESStore("wid", embedding_dim=2).publish_incremental(
            [_chunk()],
            [[0.1, 0.2]],
            replace_file_names=["source.md"],
        )

    assert client.alias_actions == []
    assert client.deleted == client.created
