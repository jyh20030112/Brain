# SimBrain

简体中文 | [English](README.md)

SimBrain 将本地资料增量写入 Elasticsearch，并提供 MCP 工具与 CLI 用于入库、查看项目、监控进度和检索。

## 快速开始

SimBrain 默认使用本机 Ollama（`http://localhost:11434`）和 `bge-m3`。请先启动 Ollama 并准备模型：

```bash
ollama pull bge-m3
```

所有使用方式都需要 Elasticsearch。至少配置连接地址和认证信息：

```dotenv
ES_URL=https://your-es-host:9200
ES_API_KEY=your_es_api_key
```

完整字段与默认值见 [.env.example](.env.example)。从源码运行时，可复制为 `.env`：

```bash
cp .env.example .env
```

`MINERU_API_TOKEN` 是可选的；配置后 PDF 优先使用 MinerU 解析，否则回退到 pypdf。

支持的资料格式：PDF、DOCX、TXT、TEXT、Markdown、CSV、XLSX。

## MCP 使用

SimBrain 通过标准输入输出（stdio）提供 MCP 服务。已发布到 PyPI 后，启动命令为：

```bash
uvx --from simbrain simbrain-mcp
```

服务提供以下工具：

| 工具 | 参数 | 用途 |
| --- | --- | --- |
| `simbrain-ingest` | `input_dir`, `output_dir`, `project` | 增量入库 |
| `simbrain-status` | `output_dir` | 列出全部 project |
| `simbrain-status-realtime` | `output_dir`, `project` | 实时查看入库进度 |
| `simbrain-search` | `question`, `project`, `top_k` | 向量与关键词融合检索 |

### Codex 配置

在 `~/.codex/config.toml` 中添加：

```toml
[mcp_servers.simbrain]
command = "uvx"
args = ["--from", "simbrain", "simbrain-mcp"]

[mcp_servers.simbrain.env]
ES_URL = "https://your-es-host:9200"
ES_API_KEY = "your_es_api_key"
MINERU_API_TOKEN = "your_mineru_token"
```

也可使用命令添加：

```bash
codex mcp add simbrain \
  --env ES_URL=https://your-es-host:9200 \
  --env ES_API_KEY=your_es_api_key \
  --env MINERU_API_TOKEN=your_mineru_token \
  -- uvx --from simbrain simbrain-mcp
```

### Claude Desktop 配置

在 `claude_desktop_config.json` 的 `mcpServers` 中添加：

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

如果不使用 MinerU，可删除 `MINERU_API_TOKEN`。其余配置使用 [.env.example](.env.example) 所定义的默认值；如需改用 OpenAI 兼容 embedding 服务，请同时设置 `EMBEDDING_PROVIDER`、`EMBEDDING_URL`、`EMBEDDING_API_KEY` 和 `EMBEDDING_MODEL`。

## CLI 使用

以下命令使用 PyPI 包；开发本仓库时，将 `uvx --from simbrain` 替换为 `uv run`。

### 增量入库

```bash
uvx --from simbrain simbrain-ingest \
  --input-dir ./docs \
  --output-dir ./output \
  --project my-knowledge-base
```

同名文件会更新，内容未变化的文件会跳过；本次未提供的历史文件会保留。

### 查看全部 project

```bash
uvx --from simbrain simbrain-status --output-dir ./output
```

### 实时查看入库进度

```bash
uvx --from simbrain simbrain-status \
  --output-dir ./output \
  --project my-knowledge-base
```

### 检索资料

```bash
uvx --from simbrain simbrain-search \
  --question "如何配置访问权限？" \
  --project my-knowledge-base \
  --top-k 10
```

`top-k` 的范围为 1–100。检索返回原始资料切片及其来源信息，不生成 LLM 回答。

## 输出与测试

CLI 的成功结果均为 JSON；实时进度命令在状态改变时输出 JSON Lines。

运行测试：

```bash
uv run pytest -q
```

如需运行 Elasticsearch 集成测试：

```bash
SIMBRAIN_TEST_ES_URL=http://localhost:9200 uv run pytest -q -m integration
```
