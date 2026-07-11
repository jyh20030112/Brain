# Brain

知识库原始资料入库与检索工具。入库和检索是两个独立 CLI，共享相同的配置、Embedding 客户端和 Elasticsearch 存储层。

## 目录结构

```text
.
├── pyproject.toml
├── src/
│   └── brain/
│       ├── cli/
│       │   ├── ingest.py              # brain-ingest 入口
│       │   └── search.py              # brain-search 入口
│       ├── documents/
│       │   ├── loaders.py             # 文件与 MinerU 解析
│       │   ├── cleaning.py            # 文本清洗
│       │   └── chunking.py            # 切片
│       ├── storage/
│       │   └── elasticsearch_store.py # ES 发布与检索
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
PROJECT=产品知识库
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
uv run brain-ingest --input-dir ./docs --project 产品知识库
```

入库流程只处理原始资料：加载与 MinerU 解析、清洗、切片、向量化、ES 原子发布。查询别名为 `docs_<workspace_id>_current`。

## 检索 CLI

人类可读输出：

```bash
uv run brain-search "这个产品怎么使用？" --top-k 5
```

JSON 输出，适合被其他程序调用：

```bash
uv run brain-search "这个产品怎么使用？" --top-k 5 --json
```

检索 CLI 使用与入库相同的 `PROJECT`、Embedding 模型和维度，通过向量召回与关键词召回进行 RRF 融合。可用 `--project` 查询其他项目：

```bash
uv run brain-search "注意事项" --project 另一个知识库 --json
```

## 测试

```bash
uv run pytest -q
```
