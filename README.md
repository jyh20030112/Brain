# SimBrain

[简体中文](README_zh-CN.md) | English

SimBrain incrementally ingests local documents into Elasticsearch and provides MCP tools and CLIs for ingestion, project inspection, progress monitoring, and retrieval.

## Quick start

SimBrain defaults to local Ollama at `http://localhost:11434` with the `bge-m3` model. Start Ollama and install the model first:

```bash
ollama pull bge-m3
```

Every usage mode requires Elasticsearch. At minimum, configure its endpoint and authentication:

```dotenv
ES_URL=https://your-es-host:9200
ES_API_KEY=your_es_api_key
```

See [.env.example](.env.example) for all fields and defaults. When running from source, copy it to `.env`:

```bash
cp .env.example .env
```

`MINERU_API_TOKEN` is optional. When configured, MinerU is preferred for PDF parsing; otherwise SimBrain falls back to pypdf.

Supported document formats: PDF, DOCX, TXT, TEXT, Markdown, CSV, and XLSX.

## Use with MCP

SimBrain exposes an MCP server over standard input/output (stdio). Once published on PyPI, start it with:

```bash
uvx --from simbrain simbrain-mcp
```

The server exposes these tools:

| Tool | Arguments | Purpose |
| --- | --- | --- |
| `simbrain-ingest` | `input_dir`, `output_dir`, `project` | Incremental ingestion |
| `simbrain-status` | `output_dir` | List all projects |
| `simbrain-status-realtime` | `output_dir`, `project` | Monitor ingestion progress |
| `simbrain-search` | `question`, `project`, `top_k` | Hybrid vector and keyword retrieval |

### Codex configuration

Add this to `~/.codex/config.toml`:

```toml
[mcp_servers.simbrain]
command = "uvx"
args = ["--from", "simbrain", "simbrain-mcp"]

[mcp_servers.simbrain.env]
ES_URL = "https://your-es-host:9200"
ES_API_KEY = "your_es_api_key"
MINERU_API_TOKEN = "your_mineru_token"
```

Or register it from the command line:

```bash
codex mcp add simbrain \
  --env ES_URL=https://your-es-host:9200 \
  --env ES_API_KEY=your_es_api_key \
  --env MINERU_API_TOKEN=your_mineru_token \
  -- uvx --from simbrain simbrain-mcp
```

### Claude Desktop configuration

Add this entry under `mcpServers` in `claude_desktop_config.json`:

```json
{
  "simbrain": {
    "command": "uvx",
    "args": ["--from", "simbrain", "simbrain-mcp"],
    "env": {
      "ES_URL": "https://your-es-host:9200",
      "ES_API_KEY": "your_es_api_key",
      "MINERU_API_TOKEN": "your_mineru_token"
    }
  }
}
```

Remove `MINERU_API_TOKEN` if MinerU is not needed. All remaining settings use the defaults from [.env.example](.env.example). To use an OpenAI-compatible embedding service, set `EMBEDDING_PROVIDER`, `EMBEDDING_URL`, `EMBEDDING_API_KEY`, and `EMBEDDING_MODEL` together.

## Use with the CLI

The examples below use the PyPI package. When developing this repository, replace `uvx --from simbrain` with `uv run`.

### Incremental ingestion

```bash
uvx --from simbrain simbrain-ingest \
  --input-dir ./docs \
  --output-dir ./output \
  --project my-knowledge-base
```

Files with the same name are updated, unchanged files are skipped, and historical files omitted from the current input are retained.

### List all projects

```bash
uvx --from simbrain simbrain-status --output-dir ./output
```

### Monitor ingestion progress

```bash
uvx --from simbrain simbrain-status \
  --output-dir ./output \
  --project my-knowledge-base
```

### Search a project

```bash
uvx --from simbrain simbrain-search \
  --question "How do I configure access permissions?" \
  --project my-knowledge-base \
  --top-k 10
```

`top-k` must be between 1 and 100. Results contain source chunks and metadata; SimBrain does not generate an LLM answer.

## Output and tests

Successful CLI calls return JSON. The progress command emits JSON Lines when its state changes.

Run tests:

```bash
uv run pytest -q
```

To include the Elasticsearch integration test:

```bash
SIMBRAIN_TEST_ES_URL=http://localhost:9200 uv run pytest -q -m integration
```
