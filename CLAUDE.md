# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Brain is a knowledge base ingestion pipeline for customer service (客服) contexts. It takes product documents (护肤品/beauty products), generates structured QA pairs via LLM + retrieval-augmented generation, and indexes everything into Elasticsearch for retrieval.

## Build & Run

- **Install dependencies**: `uv sync`
- **Run the MVP ingestion pipeline**: `uv run python mvp_ingest.py`
- **Run main**: `uv run python main.py` (currently a placeholder)

All dependencies are in `pyproject.toml`; not yet declared there explicitly but required at runtime: `openai`, `elasticsearch[async]`, `pypdf`, `python-docx`, `pandas`, `openpyxl`, `pypdfium2`, `mineru-vl-utils`.

## Architecture

### `mvp_ingest.py` — the core pipeline

A self-contained ingestion script (~1300 lines, zero project imports). The execution flow is:

1. **Document loading** (`load_docs`) — Supports PDF (MinerU OCR with pypdf fallback), DOCX, TXT/MD, CSV, XLSX. Wraps everything into `DocumentRecord` (list of `DocumentPage`).
2. **Text cleaning** (`clean_text`) — CJK-aware: normalizes whitespace, removes control chars, deduplicates consecutive identical lines, handles fullwidth spaces and non-breaking spaces.
3. **Semantic chunking** (`chunk_docs`) — Splits by paragraph, then by sentence boundaries for oversized paragraphs. Builds `TextChunk` objects with deduplication.
4. **LLM product description** (`LLMClient.generate_product_description`) — Feeds all document text to the LLM to produce a structured product intro in Markdown.
5. **Customer question generation** (`LLMClient.generate_customer_questions`) — LLM generates questions + semantic variants a real customer would ask.
6. **Hybrid retrieval** (`ESStore.search_docs`) — Two-path recall (vector kNN + BM25 keyword) fused via RRF (Reciprocal Rank Fusion).
7. **Batch QA generation** (`generate_qa_pairs`) — For each question, retrieves top-K chunks, then LLM generates an answer from evidence only. ThreadPoolExecutor-parallelized.
8. **Deduplication** (`dedupe_qa`) — Normalized question text dedup, merges evidence, keeps longest answer.
9. **ES indexing** — Two indices: `docs_{workspace_id}` (document chunks with embeddings) and `qa_{workspace_id}` (QA pairs with embeddings).
10. **Output** — `qa_list.json`, `qa_list.md`, `qa_list.csv`, `product_description.md`.

### Key design decisions

- **Sync-first**: Despite Elasticsearch using `AsyncElasticsearch`, all operations are wrapped to run synchronously via `_run_async()` which spawns a new event loop. This keeps the pipeline simple and linear.
- **Thinking mode off**: The `LLMClient` disables thinking/reasoning for GLM and Qwen models via `extra_body` to avoid extra tokens.
- **Embedding batching**: Embeddings are batched in groups of 64 with retry (3 attempts).
- **LLM evidence-only**: The QA prompt (`PROMPT_QA_EXTRACT`) strictly instructs the LLM to only use provided context, never fabricate, and use natural customer-service tone.
- **Workspace isolation**: ES indices are namespaced by `workspace_id` (MD5 hash of project name, first 16 chars).

### Configuration

All config lives in the `Config` dataclass in `mvp_ingest.py` (lines 76-105). Key parameters:
- `input_dir` — document source directory (recursive scan)
- `project` — project name, used for ES index naming
- `llm_base_url` / `llm_model` — OpenAI-compatible LLM endpoint
- `es_url` — Elasticsearch endpoint
- `mineru_server` — optional remote MinerU OCR server for PDF
- `qa_limit` / `qa_generalization` / `chunk_size` / `chunk_overlap`

### `main.py`

Currently a placeholder (`print("Hello from brain!")`). The real pipeline is in `mvp_ingest.py`.

### No tests yet

There are no test files or test framework configured. The project is in early MVP stage.
