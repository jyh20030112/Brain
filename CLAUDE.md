# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Brain is a knowledge base ingestion pipeline for customer service (客服) contexts. It takes product documents (护肤品/beauty products), generates structured QA pairs via LLM + retrieval-augmented generation, and indexes everything into Elasticsearch for retrieval.

## Build & Run

- **Install dependencies**: `uv sync`
- **Run the MVP ingestion pipeline**: `uv run python main.py`
- **Run tests**: `uv run pytest`

Runtime and dev dependencies are declared in `pyproject.toml`; `uv.lock` should be kept in sync.

## Architecture

### Package layout

`main.py` is a compatibility entrypoint only. The implementation lives in the `src` package:

- `src.config` — environment-backed `Config`
- `src.models` — shared dataclasses (`DocumentRecord`, `TextChunk`, `QAPair`, etc.)
- `src.documents` — document loading, text cleaning, and semantic chunking
- `src.llm` — OpenAI-compatible/Ollama client and prompt templates
- `src.storage` — Elasticsearch indexing and hybrid retrieval
- `src.qa` — batch QA generation and question deduplication
- `src.output` — Markdown/CSV row formatting
- `src.pipeline` — end-to-end ingestion orchestration

### Key design decisions

- **Sync-first**: Despite Elasticsearch using `AsyncElasticsearch`, all operations are wrapped to run synchronously via `_run_async()` which spawns a new event loop. This keeps the pipeline simple and linear.
- **Thinking mode off**: The `LLMClient` disables thinking/reasoning for GLM and Qwen models via `extra_body` to avoid extra tokens.
- **Embedding batching**: Embeddings are batched in groups of 64 with retry (3 attempts).
- **LLM evidence-only**: The QA prompt (`PROMPT_QA_EXTRACT`) strictly instructs the LLM to only use provided context, never fabricate, and use natural customer-service tone.
- **Workspace isolation**: ES indices are namespaced by `workspace_id` (MD5 hash of project name, first 16 chars).

### Configuration

All config lives in the `Config` dataclass in `src/config.py`. Key parameters:
- `input_dir` — document source directory (recursive scan)
- `project` — project name, used for ES index naming
- `llm_base_url` / `llm_model` — OpenAI-compatible LLM endpoint
- `es_url` — Elasticsearch endpoint
- `mineru_api_token` — optional MinerU cloud OCR token for PDF
- `qa_limit` / `qa_generalization` / `chunk_size` / `chunk_overlap`

### Tests

The `tests/` suite covers offline logic only: cleaning, chunking, QA deduplication, and output formatting. It does not call OpenAI, Ollama, MinerU, or Elasticsearch.
