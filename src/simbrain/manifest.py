from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from simbrain.project import atomic_write_json, read_json
from simbrain.utils import utc_now


SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"
PENDING_MANIFEST_NAME = ".manifest.pending.json"


def load_manifest(project_dir: Path) -> dict[str, Any] | None:
    return read_json(project_dir / MANIFEST_NAME)


def load_pending_manifest(project_dir: Path) -> dict[str, Any] | None:
    return read_json(project_dir / PENDING_MANIFEST_NAME)


def file_records_by_name(manifest: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not manifest:
        return {}
    return {
        str(item.get("file_name", "")).casefold(): dict(item)
        for item in manifest.get("files", [])
        if isinstance(item, dict) and item.get("file_name")
    }


def build_manifest(
    *,
    project: str,
    workspace_id: str,
    embedding_model: str,
    embedding_dim: int,
    active_index: str,
    index_version: str,
    inventory: list[dict[str, Any]],
    previous_manifest: dict[str, Any] | None,
    incoming_files: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    now = utc_now()
    previous = file_records_by_name(previous_manifest)
    files: list[dict[str, Any]] = []
    for indexed in inventory:
        key = str(indexed["file_name"]).casefold()
        old = previous.get(key, {})
        incoming = incoming_files.get(key, {})
        record = {
            "file_name": indexed["file_name"],
            "file_type": incoming.get("file_type") or old.get("file_type") or indexed.get("file_type") or "unknown",
            "title": incoming.get("title") or old.get("title") or indexed.get("title") or Path(indexed["file_name"]).stem,
            "source_path": incoming.get("source_path") or old.get("source_path") or indexed.get("source_path") or indexed["file_name"],
            "sha256": incoming.get("sha256") or old.get("sha256"),
            "size_bytes": incoming.get("size_bytes") if "size_bytes" in incoming else old.get("size_bytes"),
            "page_count": incoming.get("page_count") or indexed.get("page_count") or old.get("page_count") or 0,
            "chunk_count": int(indexed.get("chunk_count", 0)),
            "parser": incoming.get("parser") or old.get("parser") or indexed.get("parser") or "legacy",
            "mineru_artifact": (
                incoming.get("mineru_artifact")
                if "mineru_artifact" in incoming
                else old.get("mineru_artifact") or indexed.get("mineru_artifact")
            ),
            "first_ingested_at": old.get("first_ingested_at") or now,
            "updated_at": incoming.get("updated_at") or old.get("updated_at") or now,
        }
        files.append(record)

    files.sort(key=lambda item: item["file_name"].casefold())
    topics: list[str] = []
    normalized_topics: set[str] = set()
    for item in files:
        topic = str(item.get("title") or Path(item["file_name"]).stem).strip()[:80]
        normalized = topic.casefold()
        if topic and normalized not in normalized_topics:
            topics.append(topic)
            normalized_topics.add(normalized)
        if len(topics) == 5:
            break
    if topics:
        description = f"包含 {len(files)} 份资料，主题包括：{'、'.join(topics)}。"
    else:
        description = f"包含 {len(files)} 份资料。"

    return {
        "schema_version": SCHEMA_VERSION,
        "project": project,
        "workspace_id": workspace_id,
        "embedding_model": embedding_model,
        "embedding_dim": embedding_dim,
        "description": description,
        "topics": topics,
        "file_count": len(files),
        "chunk_count": sum(int(item["chunk_count"]) for item in files),
        "active_index": active_index,
        "index_version": index_version,
        "updated_at": now,
        "files": files,
    }


def write_pending_manifest(project_dir: Path, manifest: dict[str, Any]) -> Path:
    path = project_dir / PENDING_MANIFEST_NAME
    atomic_write_json(path, manifest)
    return path


def finalize_pending_manifest(project_dir: Path) -> Path:
    pending = project_dir / PENDING_MANIFEST_NAME
    target = project_dir / MANIFEST_NAME
    if not pending.is_file():
        raise RuntimeError("待发布 manifest 不存在")
    os.replace(pending, target)
    return target


def discard_pending_manifest(project_dir: Path) -> None:
    (project_dir / PENDING_MANIFEST_NAME).unlink(missing_ok=True)
