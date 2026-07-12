import json
from collections import Counter

import pytest

from brain.config import Config
from brain.ingestion import _recover_pending_manifest, run_ingestion
from brain.manifest import write_pending_manifest
from brain.storage.elasticsearch_store import PublishResult


class FakeEmbeddingClient:
    def embed(self, texts, progress_callback=None):
        if progress_callback:
            progress_callback(len(texts), len(texts))
        return [[0.1, 0.2] for _ in texts]


class FakeES:
    def __init__(self):
        self.files = {}
        self.version = None
        self.publish_count = 0

    def alias_indices(self):
        return [self.version] if self.version else []

    def publish_incremental(
        self,
        chunks,
        embeddings,
        *,
        replace_file_names,
        progress_callback=None,
        publishing_callback=None,
        prepare_manifest_callback=None,
        abort_manifest_callback=None,
    ):
        retained = sum(item["chunk_count"] for item in self.files.values())
        for name in replace_file_names:
            old = self.files.pop(name.casefold(), None)
            if old:
                retained -= old["chunk_count"]
        counts = Counter(chunk.file_name.casefold() for chunk in chunks)
        samples = {}
        for chunk in chunks:
            samples.setdefault(chunk.file_name.casefold(), chunk)
        for key, count in counts.items():
            chunk = samples[key]
            self.files[key] = {
                "file_name": chunk.file_name,
                "source_path": chunk.source_path,
                "file_type": chunk.metadata.get("extension", "unknown"),
                "title": chunk.metadata.get("document_title", ""),
                "parser": chunk.metadata.get("parser", "legacy"),
                "mineru_artifact": chunk.metadata.get("mineru_artifact") or None,
                "page_count": 1,
                "chunk_count": count,
            }
        if progress_callback:
            progress_callback(len(chunks), len(chunks))
        if publishing_callback:
            publishing_callback()
        self.publish_count += 1
        self.version = f"docs_test_current_v_{self.publish_count}"
        inventory = list(self.files.values())
        if prepare_manifest_callback:
            prepare_manifest_callback(inventory, "docs_test_current", self.version)
        return PublishResult(
            alias="docs_test_current",
            index_version=self.version,
            inventory=inventory,
            retained_chunks=retained,
            total_chunks=sum(item["chunk_count"] for item in inventory),
        )


def _config(input_dir, output_dir):
    return Config(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        project="my-knowledge-base",
        embedding_url="http://fake-embedding",
        embedding_model="fake-embedding",
        embedding_dim=2,
        es_url="http://fake-es",
        chunk_size=80,
        chunk_overlap=10,
    )


def _install_fakes(monkeypatch, es):
    monkeypatch.setattr("brain.ingestion.build_embedding_client", lambda cfg: FakeEmbeddingClient())
    monkeypatch.setattr("brain.ingestion.build_es_store", lambda cfg: es)


def test_incremental_ingestion_retains_files_missing_from_second_input(monkeypatch, tmp_path, capsys):
    es = FakeES()
    _install_fakes(monkeypatch, es)
    output_dir = tmp_path / "output"
    first = tmp_path / "first"
    first.mkdir()
    (first / "guide-a.txt").write_text("访问权限配置。", encoding="utf-8")
    second = tmp_path / "second"
    second.mkdir()
    (second / "guide-b.txt").write_text("部署操作指南。", encoding="utf-8")

    first_result = run_ingestion(_config(first, output_dir))
    second_result = run_ingestion(_config(second, output_dir))
    manifest = json.loads((output_dir / "my-knowledge-base" / "manifest.json").read_text(encoding="utf-8"))

    assert first_result["added"] == 1
    assert second_result["added"] == 1
    assert second_result["file_count"] == 2
    assert {item["file_name"] for item in manifest["files"]} == {"guide-a.txt", "guide-b.txt"}
    assert capsys.readouterr().out == ""


def test_same_file_name_replaces_old_chunks_and_unchanged_file_is_skipped(monkeypatch, tmp_path):
    es = FakeES()
    _install_fakes(monkeypatch, es)
    output_dir = tmp_path / "output"
    input_dir = tmp_path / "docs"
    input_dir.mkdir()
    path = input_dir / "guide.txt"
    path.write_text("旧版说明。", encoding="utf-8")
    run_ingestion(_config(input_dir, output_dir))
    path.write_text("新版说明，包含更多内容。", encoding="utf-8")

    updated = run_ingestion(_config(input_dir, output_dir))
    skipped = run_ingestion(_config(input_dir, output_dir))

    assert updated["updated"] == 1
    assert updated["file_count"] == 1
    assert skipped["skipped"] == 1
    assert es.publish_count == 2


def test_missing_manifest_is_rebuilt_from_existing_index_inventory(monkeypatch, tmp_path):
    es = FakeES()
    es.files["legacy.txt"] = {
        "file_name": "legacy.txt",
        "source_path": "legacy.txt",
        "file_type": "txt",
        "title": "历史资料",
        "parser": "legacy",
        "mineru_artifact": None,
        "page_count": 1,
        "chunk_count": 2,
    }
    es.version = "docs_test_current_v_legacy"
    _install_fakes(monkeypatch, es)
    input_dir = tmp_path / "docs"
    input_dir.mkdir()
    (input_dir / "new.txt").write_text("新增资料。", encoding="utf-8")

    run_ingestion(_config(input_dir, tmp_path / "output"))
    manifest = json.loads(
        (tmp_path / "output" / "my-knowledge-base" / "manifest.json").read_text(encoding="utf-8")
    )

    assert {item["file_name"] for item in manifest["files"]} == {"legacy.txt", "new.txt"}
    assert manifest["files"][0]["sha256"] is None


def test_duplicate_basename_in_one_input_is_rejected(monkeypatch, tmp_path):
    es = FakeES()
    _install_fakes(monkeypatch, es)
    input_dir = tmp_path / "docs"
    (input_dir / "a").mkdir(parents=True)
    (input_dir / "b").mkdir()
    (input_dir / "a" / "guide.txt").write_text("A", encoding="utf-8")
    (input_dir / "b" / "GUIDE.TXT").write_text("B", encoding="utf-8")

    with pytest.raises(ValueError, match="重复文件名"):
        run_ingestion(_config(input_dir, tmp_path / "output"))

    progress = json.loads((tmp_path / "output" / "my-knowledge-base" / "progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "failed"


def test_pending_manifest_is_recovered_only_when_alias_points_to_version(tmp_path):
    project_dir = tmp_path / "my-knowledge-base"
    write_pending_manifest(project_dir, {"project": "my-knowledge-base", "index_version": "version-1"})

    class AliasStore:
        def alias_indices(self):
            return ["version-1"]

    _recover_pending_manifest(project_dir, AliasStore())

    assert (project_dir / "manifest.json").exists()
    assert not (project_dir / ".manifest.pending.json").exists()
