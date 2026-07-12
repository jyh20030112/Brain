from __future__ import annotations

import argparse

from dotenv import load_dotenv

from brain.cli.output import emit_error, emit_json
from brain.config import Config
from brain.models import RetrievedChunk
from brain.project import validate_project_name
from brain.retrieval import SearchService
from brain.storage.elasticsearch_store import ProjectNotFoundError, RetrievalFailedError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="brain-search", description="在指定 project 中多路召回原始知识块")
    parser.add_argument("--question", required=True, help="查询问题或关键词")
    parser.add_argument("--project", required=True, help="知识库 project 名")
    parser.add_argument("--top-k", required=True, type=int, help="RRF 融合后最终返回数量，范围 1-100")
    return parser


def _result_to_dict(result: RetrievedChunk) -> dict:
    chunk = result.chunk
    return {
        "score": result.score,
        "retrieval_method": result.retrieval_method,
        "chunk": {
            "id": chunk.id,
            "workspace_id": chunk.workspace_id,
            "file_name": chunk.file_name,
            "source_path": chunk.source_path,
            "page_number": chunk.page_number,
            "section": chunk.section,
            "chunk_type": chunk.chunk_type,
            "content": chunk.content,
            "metadata": chunk.metadata,
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv()
    try:
        cfg = Config.from_env()
        cfg.project = validate_project_name(args.project)
        outcome = SearchService.from_config(cfg).search(args.question, top_k=args.top_k)
        payload = {
            "ok": True,
            "question": args.question,
            "project": cfg.project,
            "top_k": args.top_k,
            "count": len(outcome.results),
            "warnings": outcome.warnings,
            "results": [_result_to_dict(result) for result in outcome.results],
        }
        emit_json(payload, indent=2)
        return 0
    except ProjectNotFoundError as exc:
        return emit_error("project_not_found", str(exc), project=args.project)
    except RetrievalFailedError as exc:
        return emit_error("retrieval_failed", str(exc), project=args.project)
    except Exception as exc:
        return emit_error("search_failed", str(exc), project=args.project)


if __name__ == "__main__":
    raise SystemExit(main())
