from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from brain.cli.output import emit_error, emit_json
from brain.manifest import MANIFEST_NAME
from brain.progress.file_store import FileProgressStore
from brain.progress.models import IngestionJob
from brain.project import get_project_dir, read_json, validate_project_name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="brain-status", description="列出 project 清单或实时监控指定 project")
    parser.add_argument("--output-dir", required=True, help="project 产物根目录")
    parser.add_argument("--project", help="指定后持续监控该 project；不指定时列出全部 project")
    return parser


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _duration(job: IngestionJob) -> str:
    start = _parse_time(job.started_at)
    end = _parse_time(job.finished_at) or datetime.now(timezone.utc)
    if not start:
        return "unknown"
    seconds = max(0, int((end - start).total_seconds()))
    return f"{seconds}s"


def _is_stale(job: IngestionJob) -> bool:
    updated = _parse_time(job.updated_at)
    return bool(
        job.status == "running"
        and updated
        and (datetime.now(timezone.utc) - updated).total_seconds() > 30
    )


def _job_payload(job: IngestionJob) -> dict:
    stale = _is_stale(job)
    ok = not stale and job.status not in {"failed", "failed_manifest_sync", "cancelled"}
    payload = {"ok": ok, **job.to_dict()}
    payload["stage_percent"] = job.stage_percent
    payload["duration"] = _duration(job)
    payload["stale"] = stale
    if stale:
        payload["status"] = "stale"
        payload["error"] = {"code": "ingestion_stale", "message": "入库任务超过 30 秒没有心跳"}
    return payload


def _catalog(output_dir: Path) -> dict:
    projects = []
    if output_dir.is_dir():
        for project_dir in sorted((path for path in output_dir.iterdir() if path.is_dir()), key=lambda path: path.name.casefold()):
            manifest_path = project_dir / MANIFEST_NAME
            if not manifest_path.exists():
                continue
            try:
                manifest = read_json(manifest_path)
                if manifest is None:
                    continue
                projects.append(
                    {
                        "project": manifest.get("project", project_dir.name),
                        "description": manifest.get("description", ""),
                        "topics": manifest.get("topics", []),
                        "embedding_model": manifest.get("embedding_model"),
                        "embedding_dim": manifest.get("embedding_dim"),
                        "file_count": manifest.get("file_count", 0),
                        "chunk_count": manifest.get("chunk_count", 0),
                        "updated_at": manifest.get("updated_at"),
                        "active_index": manifest.get("active_index"),
                        "files": manifest.get("files", []),
                    }
                )
            except Exception as exc:
                projects.append({"project": project_dir.name, "error": str(exc), "manifest": str(manifest_path)})
    return {"ok": True, "output_dir": str(output_dir), "count": len(projects), "projects": projects}


def _monitor(output_dir: Path, project: str) -> int:
    project_dir = get_project_dir(output_dir, project)
    store = FileProgressStore(project_dir)
    previous = None
    while True:
        try:
            job = store.get_job()
        except Exception as exc:
            return emit_error("invalid_progress", str(exc))
        if job is None:
            return emit_error("progress_not_found", f"project 没有 progress.json: {project}")
        payload = _job_payload(job)
        encoded = json.dumps(job.to_dict(), ensure_ascii=False, sort_keys=True)
        if encoded != previous:
            emit_json(payload)
            previous = encoded
        if payload["stale"]:
            return 1
        if job.is_terminal:
            return 0 if job.status == "succeeded" else 1
        time.sleep(1.0)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output_dir = Path(args.output_dir).expanduser().resolve()
        if args.project:
            return _monitor(output_dir, validate_project_name(args.project))
        emit_json(_catalog(output_dir), indent=2)
        return 0
    except Exception as exc:
        return emit_error("status_failed", str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
