# Brain

知识库原始资料入库、检索与进度监控工具。三个 CLI 共享配置和基础设施，但业务入口彼此独立。

## 目录结构

```text
.
├── pyproject.toml
├── src/
│   └── brain/
│       ├── cli/
│       │   ├── ingest.py              # brain-ingest 入口
│       │   ├── search.py              # brain-search 入口
│       │   └── status.py              # brain-status 入口
│       ├── documents/
│       │   ├── loaders.py             # 文件与 MinerU 解析
│       │   ├── cleaning.py            # 文本清洗
│       │   └── chunking.py            # 切片
│       ├── storage/
│       │   ├── client.py              # ES 连接与同步桥接
│       │   └── elasticsearch_store.py # ES 发布与检索
│       ├── progress/
│       │   ├── models.py              # 入库任务模型
│       │   ├── store.py               # 进度存储接口
│       │   └── elasticsearch_store.py # 进度持久化
│       ├── config.py                  # 环境配置
│       ├── embeddings.py              # Embedding 客户端
│       ├── ingestion.py               # 入库业务流程
│       ├── retrieval.py               # 检索业务服务
│       ├── runtime.py                 # 共享依赖构造
│       └── models.py                  # 数据模型
└── tests/
```

## 配置

在项目根目录创建 `.env`：

```dotenv
PROJECT=my-knowledge-base
INPUT_DIR=./docs
OUTPUT_DIR=./mvp_output

EMBEDDING_PROVIDER=openai
EMBEDDING_URL=https://your-embedding-api/v1
EMBEDDING_API_KEY=your-key
EMBEDDING_MODEL=bge-m3
EMBEDDING_DIM=1024

ES_URL=http://localhost:9200
ES_USERNAME=
ES_PASSWORD=
ES_API_KEY=

MINERU_API_TOKEN=
CHUNK_SIZE=512
CHUNK_OVERLAP=120
```

`EMBEDDING_PROVIDER=ollama` 时，未配置 `EMBEDDING_URL` 会默认使用 `http://localhost:11434`。

## 入库 CLI

```bash
uv run brain-ingest
```

也可以覆盖本次运行的输入目录和项目名：

```bash
uv run brain-ingest --input-dir ./docs --project my-knowledge-base
```

入库流程只处理原始资料：加载与 MinerU 解析、清洗、切片、向量化、ES 原子发布。查询别名为 `docs_<workspace_id>_current`。

## 检索 CLI

人类可读输出：

```bash
uv run brain-search "如何配置访问权限？" --top-k 5
```

JSON 输出，适合被其他程序调用：

```bash
uv run brain-search "如何配置访问权限？" --top-k 5 --json
```

检索 CLI 使用与入库相同的 `PROJECT`、Embedding 模型和维度，通过向量召回与关键词召回进行 RRF 融合。可用 `--project` 查询其他项目：

```bash
uv run brain-search "注意事项" --project another-knowledge-base --json
```

## 入库进度 CLI

查看当前项目最新任务：

```bash
uv run brain-status --project my-knowledge-base
```

查看指定任务，或持续监控到任务结束：

```bash
uv run brain-status --job-id ingest_20260712_120000_a8f3c9d1
uv run brain-status --job-id ingest_20260712_120000_a8f3c9d1 --watch --interval 2
```

查看历史记录：

```bash
uv run brain-status --project my-knowledge-base --history 10
```

输出 JSON；与 `--watch` 一起使用时输出 JSON Lines：

```bash
uv run brain-status --job-id ingest_20260712_120000_a8f3c9d1 --json
uv run brain-status --job-id ingest_20260712_120000_a8f3c9d1 --watch --json
```

进度保存在独立索引 `brain_ingestion_jobs`，不会进入知识检索，也不会随文档索引发布而被覆盖。运行 `brain-ingest` 时会首先打印本次任务的 `job_id`。

## 测试

```bash
uv run pytest -q
```
