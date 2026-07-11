from contextlib import asynccontextmanager

from brain.progress.elasticsearch_store import ElasticsearchProgressStore, PROGRESS_INDEX


class FakeIndices:
    def __init__(self, client):
        self.client = client

    async def exists(self, *, index):
        return self.client.index_exists

    async def create(self, *, index, mappings):
        self.client.index_exists = True
        self.client.created_index = index


class FakeES:
    def __init__(self):
        self.index_exists = False
        self.created_index = None
        self.documents = {}
        self.indices = FakeIndices(self)

    def options(self, **kwargs):
        return self

    async def index(self, *, index, id, document, refresh):
        self.documents[id] = dict(document)
        return {"result": "created"}

    async def update(self, *, index, id, doc, refresh):
        self.documents[id].update(doc)
        return {"result": "updated"}

    async def get(self, *, index, id):
        if id not in self.documents:
            return {"found": False}
        return {"found": True, "_source": self.documents[id]}

    async def search(self, *, index, size, query, sort):
        workspace_id = query["term"]["workspace_id"]
        docs = [doc for doc in self.documents.values() if doc["workspace_id"] == workspace_id]
        docs.sort(key=lambda doc: doc["started_at"], reverse=True)
        return {"hits": {"hits": [{"_source": doc} for doc in docs[:size]]}}


def test_progress_store_creates_updates_and_reads_job(monkeypatch):
    client = FakeES()

    @asynccontextmanager
    async def fake_context(**kwargs):
        yield client

    monkeypatch.setattr("brain.progress.elasticsearch_store.es_context", fake_context)
    store = ElasticsearchProgressStore(es_url="http://fake-es")

    created = store.create_job(project="my-knowledge-base", workspace_id="wid")
    store.update_job(created.job_id, stage="embedding", current=2, total=4, chunks_total=4)
    store.complete_job(created.job_id, active_index="docs_wid_current")
    loaded = store.get_job(created.job_id)

    assert client.created_index == PROGRESS_INDEX
    assert loaded is not None
    assert loaded.status == "succeeded"
    assert loaded.stage == "completed"
    assert loaded.active_index == "docs_wid_current"
    assert store.list_jobs(workspace_id="wid", limit=10)[0].job_id == created.job_id
