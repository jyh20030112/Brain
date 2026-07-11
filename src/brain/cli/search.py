from __future__ import annotations

import argparse
import json
import sys

from dotenv import load_dotenv

from brain.config import Config
from brain.models import RetrievedChunk
from brain.retrieval import SearchService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="brain-search", description="从已入库知识库召回相关原始资料块")
    parser.add_argument("query", help="要查询的问题或关键词")
    parser.add_argument("--project", help="知识库项目名；默认读取 PROJECT")
    parser.add_argument("--top-k", type=int, default=5, help="返回结果数量，默认 5")
    parser.add_argument("--json", action="store_true", dest="as_json", help="以 JSON 输出，适合程序调用")
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


def _print_human(results: list[RetrievedChunk]) -> None:
    if not results:
        print("未检索到相关资料。")
        return
    for index, result in enumerate(results, start=1):
        chunk = result.chunk
        page = chunk.page_number if chunk.page_number is not None else "未知"
        print(f"[{index}] {chunk.file_name} / 第{page}页 / {chunk.section}")
        print(f"    method={result.retrieval_method} score={result.score:.6f}")
        print(f"    {chunk.content}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv()
    try:
        cfg = Config.from_env()
        if args.project:
            cfg.project = args.project
        results = SearchService.from_config(cfg).search(args.query, top_k=args.top_k)
    except Exception as exc:
        print(f"检索失败: {exc}", file=sys.stderr)
        return 1

    if args.as_json:
        payload = {"query": args.query, "count": len(results), "results": [_result_to_dict(r) for r in results]}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        _print_human(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
