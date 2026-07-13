# Brain

[简体中文](README_zh-CN.md) | English

Brain is an MCP-oriented toolkit for incremental knowledge-base ingestion, hybrid retrieval, and ingestion status inspection. It exposes three CLIs with four fixed functions. Successful command output is JSON, while live progress is emitted as JSON Lines.

## Project Structure

```text
Brain/
├── README.md                         # English documentation
├── README_zh-CN.md                   # Simplified Chinese documentation
├── pyproject.toml                    # Package metadata, dependencies, CLI entries
├── uv.lock                           # Locked dependency versions
├── src/
│   └── brain/
│       ├── __init__.py
│       ├── cli/
│       │   ├── __init__.py
│       │   ├── ingest.py             # brain-ingest
│       │   ├── search.py             # brain-search
│       │   └── status.py             # brain-status
│       ├── serve/
│       │   ├── __init__.py
│       │   ├── __main__.py            # brain-mcp entry point
│       │   └── server.py              # FastMCP server and four tools
│       ├── documents/
│       │   ├── __init__.py
│       │   ├── loaders.py            # File loading and MinerU parsing
│       │   ├── cleaning.py           # Text normalization and cleaning
│       │   └── chunking.py           # Token-budget-aware chunking
│       ├── progress/
│       │   ├── __init__.py
│       │   ├── file_store.py         # Atomic progress.json storage and heartbeat
│       │   └── models.py             # Ingestion job model
│       ├── storage/
│       │   ├── __init__.py
│       │   ├── client.py             # Elasticsearch connection helpers
│       │   └── elasticsearch_store.py # Incremental publishing and retrieval
│       ├── config.py                 # Environment-based service configuration
│       ├── constants.py              # Supported file extensions
│       ├── embeddings.py             # OpenAI-compatible and Ollama embeddings
│       ├── ingestion.py              # Incremental ingestion orchestration
│       ├── manifest.py               # Project manifest generation and recovery
│       ├── models.py                 # Documents, chunks, and retrieval models
│       ├── project.py                # Project validation, locking, atomic JSON
│       ├── retrieval.py              # Hybrid retrieval service
│       ├── runtime.py                # Runtime dependency construction
│       └── utils.py                  # Shared utility functions
└── tests/
    ├── test_chunking.py
    ├── test_cleaning.py
    ├── test_cli.py
    ├── test_elasticsearch_store.py
    ├── test_ingestion.py
    ├── test_loaders.py
    ├── test_manifest.py
    ├── test_models.py
    ├── test_progress_store.py
    ├── test_project.py
    └── test_retrieval.py
```

## Configuration

Service connections and processing settings are loaded from `.env`:

```dotenv
EMBEDDING_PROVIDER=openai
EMBEDDING_URL=https://your-embedding-api/v1
EMBEDDING_API_KEY=your-key
EMBEDDING_MODEL=bge-m3
EMBEDDING_DIM=1024

ES_URL=http://localhost:9200
ES_USERNAME=
ES_PASSWORD=
ES_API_KEY=
ES_INDEX_VERSIONS_TO_KEEP=2

MINERU_API_TOKEN=
CHUNK_SIZE=512
CHUNK_OVERLAP=120
```

`input-dir`, `output-dir`, and `project` are never filled from environment variables. They must be passed explicitly to the relevant CLI.

## Function 1: Incremental Ingestion

```bash
uv run brain-ingest \
  --input-dir ./docs \
  --output-dir ./mvp_output \
  --project my-knowledge-base
```

All three arguments are required. File identity is the case-insensitive basename:

- New filenames are appended to the project.
- Existing filenames replace all old chunks for that file.
- Unchanged files are skipped by SHA-256.
- Existing files absent from the current input remain in the project.
- Duplicate basenames in one input batch fail the entire ingestion.
- Only one ingestion may run for the same project on the same host, even when different output directories are passed.

Documents are published through a staging index and an atomic Elasticsearch alias switch. Standard output contains only the final JSON result; operational logs go to standard error.
After a successful switch, old physical index versions are removed according to `ES_INDEX_VERSIONS_TO_KEEP` (default: current plus one rollback version).

Before skipping unchanged files, ingestion verifies that the manifest's physical index is still behind the active alias and that the chunk count matches. If repair is needed, all historical files must be present in the current input so the project can be rebuilt without losing data.

Project artifacts are stored as follows:

```text
mvp_output/
└── my-knowledge-base/
    ├── manifest.json                 # Current project inventory
    ├── progress.json                 # Current or latest ingestion status
    ├── .ingest.lock                  # Per-project ingestion lock
    └── mineru/
        └── guide_<hash>/
            └── mineru_result.md      # MinerU intermediate Markdown
```

`manifest.json` describes the current Elasticsearch alias contents, including a deterministic project description, topics, file hashes, parser names, page counts, and chunk counts. No LLM is used to generate this description.

## Function 2: List All Projects

```bash
uv run brain-status --output-dir ./mvp_output
```

This scans every valid `manifest.json` below the output directory and returns project descriptions and complete file inventories as JSON. A malformed manifest is returned as an error entry without blocking other projects.

## Function 3: Monitor Ingestion Progress

```bash
uv run brain-status \
  --output-dir ./mvp_output \
  --project my-knowledge-base
```

While ingestion is running, this command reads `progress.json` once per second and emits a JSON Line only when the state changes. It exits when the task succeeds, fails, or becomes stale after 30 seconds without a heartbeat.

Stages:

```text
recovering → scanning → parsing → cleaning → chunking
→ embedding → indexing → publishing → completed
```

## Function 4: Hybrid Retrieval

```bash
uv run brain-search \
  --question "How do I configure access permissions?" \
  --project my-knowledge-base \
  --top-k 10
```

All three arguments are required, and output is always JSON. Retrieval combines vector and keyword candidates with Reciprocal Rank Fusion (RRF):

- `top-k` is the final result count and must be between 1 and 100.
- Each route retrieves `max(2 × top-k, 20)` candidates.
- Results include the filename, source path, page, section, original text, retrieval method, and RRF score.
- A missing project returns `project_not_found`. If one route fails, successful output contains a `warnings` array; if both routes fail, the command returns `retrieval_failed` with a non-zero exit code.
- The command retrieves original chunks only; it does not generate an LLM answer.

## MCP Server

Brain also exposes the CLI capabilities through a FastMCP server. Start the local
server with the standard input/output transport:

```bash
uv run brain-mcp
```

It provides four tools, with the same input and output semantics as the CLIs:

- `brain-ingest(input_dir, output_dir, project)`: incremental ingestion.
- `brain-status(output_dir)`: list every project.
- `brain-status-realtime(output_dir, project)`: monitor an ingestion task. Each changed state is sent through MCP logging and progress notifications; the final response contains all observed events.
- `brain-search(question, project, top_k)`: hybrid retrieval for a project.

## Supported Files

```text
PDF, DOCX, TXT, TEXT, Markdown, CSV, XLSX
```

When configured, MinerU is preferred for PDF parsing; failures fall back to pypdf. MinerU Markdown is stored under the corresponding project's `mineru/` directory.

## Upgrade Notes

An existing `docs_<workspace>_current` alias is copied into the new staging version during the first incremental run. The old Elasticsearch progress index is no longer used, and legacy `mvp_output/mineru_*` directories are not moved automatically.

## Tests

```bash
uv run pytest -q
```

Unit tests use isolated stores. To additionally run the live Elasticsearch alias, bulk, retrieval, and version-retention integration test:

```bash
BRAIN_TEST_ES_URL=http://localhost:9200 uv run pytest -q -m integration
```
