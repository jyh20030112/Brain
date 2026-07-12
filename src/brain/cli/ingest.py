from __future__ import annotations

import argparse

from dotenv import load_dotenv

from brain.cli.output import emit_error, emit_json
from brain.config import Config
from brain.ingestion import run_ingestion
from brain.project import ProjectLockedError, validate_project_name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="brain-ingest", description="增量解析原始资料并写入知识库")
    parser.add_argument("--input-dir", required=True, help="本次新增或更新的资料目录")
    parser.add_argument("--output-dir", required=True, help="project 清单、进度和 MinerU 产物根目录")
    parser.add_argument("--project", required=True, help="知识库 project 名")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv()
    try:
        cfg = Config.from_env()
        cfg.input_dir = args.input_dir
        cfg.output_dir = args.output_dir
        cfg.project = validate_project_name(args.project)
        emit_json(run_ingestion(cfg), indent=2)
        return 0
    except ProjectLockedError as exc:
        return emit_error("project_locked", str(exc))
    except Exception as exc:
        return emit_error("ingestion_failed", str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
