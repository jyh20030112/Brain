from __future__ import annotations

import argparse

from dotenv import load_dotenv

from brain.config import Config
from brain.ingestion import run_ingestion


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="brain-ingest", description="解析原始资料并写入知识库")
    parser.add_argument("--input-dir", help="待入库资料目录；默认读取 INPUT_DIR")
    parser.add_argument("--output-dir", help="MinerU 中间产物目录；默认读取 OUTPUT_DIR")
    parser.add_argument("--project", help="知识库项目名；默认读取 PROJECT")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv()
    cfg = Config.from_env()
    if args.input_dir:
        cfg.input_dir = args.input_dir
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.project:
        cfg.project = args.project
    run_ingestion(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
