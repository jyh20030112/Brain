import os
from uuid import uuid4

import pytest

from simbrain.models import TextChunk
from simbrain.storage.client import es_context, run_async
from simbrain.storage.elasticsearch_store import ESStore, _docs_index


pytestmark = pytest.mark.integration


@pytest.fixture
def live_store():
    url = os.getenv("SIMBRAIN_TEST_ES_URL")
    if not url:
        pytest.skip("set SIMBRAIN_TEST_ES_URL to run live Elasticsearch tests")
    workspace_id = f"integration_{uuid4().hex[:12]}"
    store = ESStore(
        workspace_id,
        es_url=url,
        es_user=os.getenv("SIMBRAIN_TEST_ES_USERNAME", ""),
        es_pass=os.getenv("SIMBRAIN_TEST_ES_PASSWORD", ""),
        es_api_key=os.getenv("SIMBRAIN_TEST_ES_API_KEY", ""),
        embedding_dim=2,
        index_versions_to_keep=1,
    )
    yield store

    async def cleanup():
        async with es_context(
            url=url,
            username=os.getenv("SIMBRAIN_TEST_ES_USERNAME", ""),
            password=os.getenv("SIMBRAIN_TEST_ES_PASSWORD", ""),
            api_key=os.getenv("SIMBRAIN_TEST_ES_API_KEY", ""),
        ) as es:
            await es.options(ignore_status=404).indices.delete(index=f"{_docs_index(workspace_id)}_v_*")

    run_async(cleanup())


def test_live_publish_alias_search_and_version_cleanup(live_store):
    chunk = TextChunk(
        id="chunk_integration",
        workspace_id=live_store.wid,
        file_name="guide.md",
        source_path="guide.md",
        content="configure access permissions",
        page_number=1,
        section="Access",
        chunk_type="paragraph",
        metadata={"extension": "md", "document_title": "Access", "document_page_count": "1"},
    )

    first = live_store.publish_incremental(
        [chunk],
        [[1.0, 0.0]],
        replace_file_names=[chunk.file_name],
    )
    second = live_store.publish_incremental(
        [chunk],
        [[1.0, 0.0]],
        replace_file_names=[chunk.file_name],
    )
    state = live_store.active_index_state()
    outcome = live_store.search_docs("access permissions", [1.0, 0.0], top_k=1)

    assert first.total_chunks == second.total_chunks == 1
    assert state["chunk_count"] == 1
    assert len(state["indices"]) == 1
    assert outcome.results[0].chunk.id == chunk.id
