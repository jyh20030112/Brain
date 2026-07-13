import time

from simbrain.progress.file_store import FileProgressStore


def test_file_progress_store_creates_updates_and_completes_job(tmp_path):
    store = FileProgressStore(tmp_path / "my-knowledge-base")

    created = store.create_job(project="my-knowledge-base", workspace_id="wid")
    store.update_job(created.job_id, stage="embedding", current=2, total=4, chunks_total=4)
    store.complete_job(created.job_id, active_index="docs_wid_current")
    loaded = store.get_job()

    assert loaded is not None
    assert loaded.status == "succeeded"
    assert loaded.stage == "completed"
    assert loaded.active_index == "docs_wid_current"


def test_file_progress_store_heartbeat_updates_timestamp(tmp_path):
    store = FileProgressStore(tmp_path / "my-knowledge-base")
    job = store.create_job(project="my-knowledge-base", workspace_id="wid")
    before = job.updated_at

    with store.heartbeat(job.job_id, interval=0.01):
        time.sleep(0.03)

    assert store.get_job().updated_at > before
