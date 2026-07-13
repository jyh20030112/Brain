"""基于 FastMCP 暴露 Brain CLI 的 MCP 工具。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastmcp import Context, FastMCP

from brain.cli.search import _result_to_dict
from brain.cli.status import _catalog, _job_payload
from brain.config import Config
from brain.ingestion import run_ingestion
from brain.progress.file_store import FileProgressStore
from brain.project import ProjectLockedError, get_project_dir, validate_project_name
from brain.retrieval import SearchService
from brain.storage.elasticsearch_store import ProjectNotFoundError, RetrievalFailedError


mcp = FastMCP(
    name="Brain",
    instructions="用于知识库的增量入库、项目查询、实时进度查看与混合检索。",
)


def _error(code: str, message: str, **details: Any) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message}
    if details:
        error["details"] = details
    return {"ok": False, "error": error}


def _load_config() -> Config:
    load_dotenv()
    return Config.from_env()


@mcp.tool(
    name="brain-ingest",
    description="增量解析目录中的资料并发布到指定知识库 project。",
    annotations={"destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
)
def brain_ingest(input_dir: str, output_dir: str, project: str) -> dict[str, Any]:
    """增量解析资料并写入指定知识库。

    Args:
        input_dir: 本次需要新增或更新的原始资料目录。
        output_dir: 保存 project 清单、进度及 MinerU 中间产物的根目录。
        project: 知识库 project 名称。
    """
    try:
        config = _load_config()
        config.input_dir = input_dir
        config.output_dir = output_dir
        config.project = validate_project_name(project)
        return run_ingestion(config)
    except ProjectLockedError as exc:
        return _error("project_locked", str(exc))
    except Exception as exc:
        return _error("ingestion_failed", str(exc))


@mcp.tool(
    name="brain-status",
    description="列出输出目录下全部知识库 project 及其完整资料清单。",
    annotations={"readOnlyHint": True, "idempotentHint": True, "openWorldHint": False},
)
def brain_status(output_dir: str) -> dict[str, Any]:
    """列出输出目录下所有知识库 project 的完整清单。

    Args:
        output_dir: 保存所有 project 产物的根目录。
    """
    try:
        return _catalog(Path(output_dir).expanduser().resolve())
    except Exception as exc:
        return _error("status_failed", str(exc))


@mcp.tool(
    name="brain-status-realtime",
    description="实时监控指定 project 的入库进度，并通过 MCP 进度与日志通知推送变化。",
    annotations={"readOnlyHint": True, "idempotentHint": True, "openWorldHint": False},
)
async def brain_status_realtime(output_dir: str, project: str, ctx: Context) -> dict[str, Any]:
    """持续监控入库任务，直到结束或超过心跳时限。

    Args:
        output_dir: 保存 project 产物的根目录。
        project: 要监控的知识库 project 名称。
        ctx: FastMCP 注入的上下文，用于发送状态日志和进度通知。
    """
    try:
        project = validate_project_name(project)
        resolved_output_dir = Path(output_dir).expanduser().resolve()
        store = FileProgressStore(get_project_dir(resolved_output_dir, project))
        previous: str | None = None
        events: list[dict[str, Any]] = []

        while True:
            # 进度文件读取很短，直接执行可避免为每秒轮询额外创建线程。
            job = store.get_job()
            if job is None:
                return _error("progress_not_found", f"project 没有 progress.json: {project}")

            payload = _job_payload(job)
            encoded = json.dumps(job.to_dict(), ensure_ascii=False, sort_keys=True)
            if encoded != previous:
                previous = encoded
                events.append(payload)
                await ctx.info(json.dumps(payload, ensure_ascii=False))
                await ctx.report_progress(progress=payload["stage_percent"] or 0, total=100)

            if payload["stale"] or job.is_terminal:
                return {
                    "ok": payload["ok"],
                    "output_dir": str(resolved_output_dir),
                    "project": project,
                    "events": events,
                    "final": payload,
                }
            await asyncio.sleep(1.0)
    except Exception as exc:
        return _error("status_failed", str(exc))


@mcp.tool(
    name="brain-search",
    description="在指定 project 中执行向量与关键词融合检索，返回原始知识块。",
    annotations={"readOnlyHint": True, "idempotentHint": True, "openWorldHint": False},
)
def brain_search(question: str, project: str, top_k: int) -> dict[str, Any]:
    """在指定知识库中执行向量与关键词融合检索。

    Args:
        question: 查询问题或关键词。
        project: 要检索的知识库 project 名称。
        top_k: 融合后返回的结果数，取值范围为 1 至 100。
    """
    try:
        config = _load_config()
        config.project = validate_project_name(project)
        outcome = SearchService.from_config(config).search(question, top_k=top_k)
        return {
            "ok": True,
            "question": question,
            "project": config.project,
            "top_k": top_k,
            "count": len(outcome.results),
            "warnings": outcome.warnings,
            "results": [_result_to_dict(result) for result in outcome.results],
        }
    except ProjectNotFoundError as exc:
        return _error("project_not_found", str(exc), project=project)
    except RetrievalFailedError as exc:
        return _error("retrieval_failed", str(exc), project=project)
    except Exception as exc:
        return _error("search_failed", str(exc), project=project)
