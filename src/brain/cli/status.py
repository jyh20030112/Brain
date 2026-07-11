from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone

from dotenv import load_dotenv

from brain.config import Config
from brain.progress.models import IngestionJob
from brain.runtime import build_progress_store


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="brain-status", description="查询和实时监控知识库入库任务")
    parser.add_argument("--job-id", help="指定入库任务 ID；不指定时查询项目最新任务")
    parser.add_argument("--project", help="知识库项目名；默认读取 PROJECT")
    parser.add_argument("--history", type=int, metavar="N", help="查看最近 N 次入库任务")
    parser.add_argument("--watch", action="store_true", help="持续刷新，直到任务结束")
    parser.add_argument("--interval", type=float, default=2.0, help="watch 刷新间隔秒数，默认 2")
    parser.add_argument("--json", action="store_true", dest="as_json", help="输出 JSON；watch 时输出 JSON Lines")
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
        return "未知"
    seconds = max(0, int((end - start).total_seconds()))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _is_stale(job: IngestionJob, *, threshold_seconds: int = 30) -> bool:
    updated = _parse_time(job.updated_at)
    if job.status != "running" or not updated:
        return False
    return (datetime.now(timezone.utc) - updated).total_seconds() > threshold_seconds


def _job_payload(job: IngestionJob) -> dict:
    payload = job.to_dict()
    payload["stage_percent"] = job.stage_percent
    payload["duration"] = _duration(job)
    payload["stale"] = _is_stale(job)
    return payload


def _print_job(job: IngestionJob) -> None:
    status = "stale" if _is_stale(job) else job.status
    progress = f"{job.current} / {job.total}"
    if job.stage_percent is not None:
        progress += f" ({job.stage_percent:.1f}%)"
    print(f"任务: {job.job_id}")
    print(f"项目: {job.project}")
    print(f"状态: {status.upper()}")
    print(f"阶段: {job.stage}")
    print(f"进度: {progress}")
    print(
        f"文档: 成功 {job.documents_succeeded} / "
        f"失败 {job.documents_failed} / 总计 {job.documents_total}"
    )
    print(f"Chunks: {job.chunks_total}")
    print(f"运行时间: {_duration(job)}")
    print(f"最后更新: {job.updated_at}")
    if job.active_index:
        print(f"活跃索引: {job.active_index}")
    if job.error:
        print(f"错误: {job.error}")


def _print_history(jobs: list[IngestionJob], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps({"count": len(jobs), "jobs": [_job_payload(job) for job in jobs]}, ensure_ascii=False, indent=2))
        return
    if not jobs:
        print("没有找到入库任务。")
        return
    for job in jobs:
        print(
            f"{job.job_id}  {job.status:<9}  {job.stage:<10}  "
            f"docs={job.documents_succeeded}/{job.documents_total}  "
            f"chunks={job.chunks_total}  duration={_duration(job)}"
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.history is not None and args.history <= 0:
        print("查询失败: --history 必须大于 0", file=sys.stderr)
        return 2
    if args.interval < 0.2:
        print("查询失败: --interval 不能小于 0.2 秒", file=sys.stderr)
        return 2
    if args.history is not None and args.watch:
        print("查询失败: --history 不能与 --watch 同时使用", file=sys.stderr)
        return 2

    load_dotenv()
    try:
        cfg = Config.from_env()
        if args.project:
            cfg.project = args.project
        cfg.validate_for_status()
        store = build_progress_store(cfg)

        if args.history is not None:
            jobs = store.list_jobs(workspace_id=cfg.workspace_id, limit=args.history)
            _print_history(jobs, as_json=args.as_json)
            return 0

        job = store.get_job(args.job_id) if args.job_id else None
        if job is None and not args.job_id:
            jobs = store.list_jobs(workspace_id=cfg.workspace_id, limit=1)
            job = jobs[0] if jobs else None
        if job is None:
            print("没有找到入库任务。", file=sys.stderr)
            return 1

        watched_job_id = job.job_id
        while True:
            if args.as_json:
                print(json.dumps(_job_payload(job), ensure_ascii=False), flush=True)
            else:
                _print_job(job)
            if not args.watch or job.is_terminal:
                return 0
            time.sleep(args.interval)
            refreshed = store.get_job(watched_job_id)
            if refreshed is None:
                print(f"任务已不存在: {watched_job_id}", file=sys.stderr)
                return 1
            job = refreshed
            if not args.as_json:
                print("---")
    except Exception as exc:
        print(f"查询失败: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
