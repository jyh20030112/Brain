from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Any

from brain.config import Config
from brain.constants import SUPPORTED_EXTENSIONS
from brain.documents.chunking import chunk_docs
from brain.documents.cleaning import clean_docs
from brain.documents.loaders import load_docs
from brain.manifest import (
    build_manifest,
    discard_pending_manifest,
    file_records_by_name,
    finalize_pending_manifest,
    load_manifest,
    load_pending_manifest,
    write_pending_manifest,
)
from brain.progress.file_store import FileProgressStore, utc_now
from brain.project import ProjectLock, get_project_dir, sha256_file
from brain.runtime import build_embedding_client, build_es_store


class ManifestSyncError(RuntimeError):
    pass


def _log(message: str) -> None:
    print(message, file=sys.stderr)


def _scan_files(input_dir: Path, output_root: Path) -> list[Path]:
    files = []
    excluded_root = output_root if output_root.is_relative_to(input_dir) else None
    for path in input_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        resolved = path.resolve()
        if excluded_root and resolved.is_relative_to(excluded_root):
            continue
        files.append(path)
    files.sort(key=lambda path: (path.name.casefold(), str(path)))
    duplicate_names = [name for name, count in Counter(path.name.casefold() for path in files).items() if count > 1]
    if duplicate_names:
        raise ValueError(f"输入目录存在重复文件名: {', '.join(sorted(duplicate_names))}")
    return files


def _recover_pending_manifest(project_dir: Path, es) -> None:
    pending = load_pending_manifest(project_dir)
    if not pending:
        return
    if pending.get("index_version") in es.alias_indices():
        finalize_pending_manifest(project_dir)
    else:
        discard_pending_manifest(project_dir)


def run_ingestion(cfg: Config) -> dict[str, Any]:
    cfg.validate_for_ingestion()
    input_dir = Path(cfg.input_dir).expanduser().resolve()
    output_root = Path(cfg.output_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise ValueError(f"input-dir 路径不存在: {input_dir}")
    if input_dir == output_root:
        raise ValueError("input-dir 与 output-dir 不能是同一目录")

    project_dir = get_project_dir(output_root, cfg.project)
    with ProjectLock(project_dir):
        progress = FileProgressStore(project_dir)
        job = progress.create_job(project=cfg.project, workspace_id=cfg.workspace_id)
        es = build_es_store(cfg)
        try:
            with progress.heartbeat(job.job_id):
                progress.update_job(job.job_id, stage="recovering", current=0, total=1)
                _recover_pending_manifest(project_dir, es)
                result = _execute_ingestion(cfg, input_dir, output_root, project_dir, progress, job.job_id, es)
                progress.complete_job(job.job_id, active_index=result["active_index"])
                return {"ok": True, "job_id": job.job_id, **result}
        except ManifestSyncError as exc:
            progress.fail_job(job.job_id, error=str(exc), status="failed_manifest_sync")
            raise
        except BaseException as exc:
            try:
                progress.fail_job(job.job_id, error=f"{type(exc).__name__}: {exc}")
            except Exception as progress_exc:
                _log(f"[进度] 无法记录失败状态: {progress_exc}")
            raise


def _execute_ingestion(
    cfg: Config,
    input_dir: Path,
    output_root: Path,
    project_dir: Path,
    progress: FileProgressStore,
    job_id: str,
    es,
) -> dict[str, Any]:
    _log(f"[输入] 扫描 {input_dir}")
    file_paths = _scan_files(input_dir, output_root)
    if not file_paths:
        raise ValueError("未找到受支持的文档文件")

    progress.update_job(job_id, stage="scanning", current=len(file_paths), total=len(file_paths), documents_total=len(file_paths))
    previous_manifest = load_manifest(project_dir)
    if previous_manifest:
        previous_model = previous_manifest.get("embedding_model")
        previous_dim = previous_manifest.get("embedding_dim")
        if previous_model and previous_model != cfg.embedding_model:
            raise ValueError("同一 project 增量入库必须使用相同的 EMBEDDING_MODEL")
        if previous_dim and int(previous_dim) != cfg.embedding_dim:
            raise ValueError("同一 project 增量入库必须使用相同的 EMBEDDING_DIM")
    previous_files = file_records_by_name(previous_manifest)
    hashes = {path.name.casefold(): sha256_file(path) for path in file_paths}
    added_paths: list[Path] = []
    updated_paths: list[Path] = []
    skipped_paths: list[Path] = []
    for path in file_paths:
        previous = previous_files.get(path.name.casefold())
        if previous and previous.get("sha256") == hashes[path.name.casefold()]:
            skipped_paths.append(path)
        elif previous:
            updated_paths.append(path)
        else:
            added_paths.append(path)
    changed_paths = added_paths + updated_paths
    progress.update_job(
        job_id,
        files_added=len(added_paths),
        files_updated=len(updated_paths),
        files_skipped=len(skipped_paths),
    )

    if not changed_paths:
        if not previous_manifest:
            raise RuntimeError("没有可处理的新文件，且 project manifest 不存在")
        active_index = str(previous_manifest.get("active_index") or f"docs_{cfg.workspace_id}_current")
        return {
            "project": cfg.project,
            "added": 0,
            "updated": 0,
            "skipped": len(skipped_paths),
            "file_count": int(previous_manifest.get("file_count", 0)),
            "chunk_count": int(previous_manifest.get("chunk_count", 0)),
            "active_index": active_index,
            "manifest": str(project_dir / "manifest.json"),
        }

    embeddings = build_embedding_client(cfg)
    _log(f"[解析] 处理 {len(changed_paths)} 个新增或更新文件")
    progress.update_job(job_id, stage="parsing", current=0, total=len(changed_paths))
    docs = load_docs(
        changed_paths,
        mineru_api_token=cfg.mineru_api_token,
        output_dir=project_dir,
        source_root=input_dir,
        progress_callback=lambda current, total, name: progress.update_job(
            job_id,
            stage="parsing",
            current=current,
            total=total,
            current_file=name,
            documents_succeeded=current,
        ),
    )
    if not docs:
        raise RuntimeError("没有可处理的文档")

    progress.update_job(job_id, stage="cleaning", current=0, total=len(docs), current_file=None)
    docs = clean_docs(docs)
    progress.update_job(job_id, stage="cleaning", current=len(docs), total=len(docs))

    progress.update_job(job_id, stage="chunking", current=0, total=len(docs))
    chunks = chunk_docs(docs, workspace_id=cfg.workspace_id, chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap)
    if not chunks:
        raise RuntimeError("切片结果为空")
    chunk_counts = Counter(chunk.file_name.casefold() for chunk in chunks)
    progress.update_job(job_id, stage="chunking", current=len(docs), total=len(docs), chunks_total=len(chunks))

    now = utc_now()
    paths_by_name = {path.name.casefold(): path for path in changed_paths}
    incoming_files: dict[str, dict[str, Any]] = {}
    for doc in docs:
        key = doc.file_name.casefold()
        path = paths_by_name[key]
        incoming_files[key] = {
            "file_name": doc.file_name,
            "file_type": doc.file_type,
            "title": doc.title,
            "source_path": doc.source_path,
            "sha256": hashes[key],
            "size_bytes": path.stat().st_size,
            "page_count": len(doc.pages),
            "chunk_count": chunk_counts[key],
            "parser": doc.metadata.get("parser", doc.file_type),
            "mineru_artifact": doc.metadata.get("mineru_artifact") or None,
            "updated_at": now,
        }

    progress.update_job(job_id, stage="embedding", current=0, total=len(chunks))
    vectors = embeddings.embed(
        [chunk.embedding_text for chunk in chunks],
        progress_callback=lambda current, total: progress.update_job(
            job_id, stage="embedding", current=current, total=total
        ),
    )
    progress.update_job(job_id, stage="indexing", current=0, total=len(chunks))
    prepared_manifest: dict[str, Any] = {}

    def _prepare_manifest(inventory: list[dict[str, Any]], alias: str, index_version: str) -> None:
        candidate = build_manifest(
            project=cfg.project,
            workspace_id=cfg.workspace_id,
            embedding_model=cfg.embedding_model,
            embedding_dim=cfg.embedding_dim,
            active_index=alias,
            index_version=index_version,
            inventory=inventory,
            previous_manifest=previous_manifest,
            incoming_files=incoming_files,
        )
        write_pending_manifest(project_dir, candidate)
        prepared_manifest.update(candidate)

    publish_result = es.publish_incremental(
        chunks,
        vectors,
        replace_file_names=[path.name for path in changed_paths],
        progress_callback=lambda current, total: progress.update_job(
            job_id, stage="indexing", current=current, total=total
        ),
        publishing_callback=lambda: progress.update_job(
            job_id, stage="publishing", current=0, total=1, current_file=None
        ),
        prepare_manifest_callback=_prepare_manifest,
        abort_manifest_callback=lambda: discard_pending_manifest(project_dir),
    )
    try:
        manifest_path = finalize_pending_manifest(project_dir)
    except Exception as exc:
        raise ManifestSyncError(f"ES 已发布，但 manifest 同步失败: {exc}") from exc

    return {
        "project": cfg.project,
        "added": len(added_paths),
        "updated": len(updated_paths),
        "skipped": len(skipped_paths),
        "file_count": int(prepared_manifest.get("file_count", 0)),
        "chunk_count": publish_result.total_chunks,
        "active_index": publish_result.alias,
        "manifest": str(manifest_path),
    }
