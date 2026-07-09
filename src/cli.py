from __future__ import annotations

from dotenv import load_dotenv

from src.config import Config
from src.pipeline import run_pipeline


def main() -> None:
    load_dotenv()
    cfg = Config.from_env()
    run_pipeline(cfg)
