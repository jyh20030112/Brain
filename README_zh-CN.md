# Brain

简体中文 | [English](README.md)

Brain 是一个面向 MCP 服务的知识库原始资料增量入库、多路召回与状态查询工具。项目只提供 3 个 CLI、4 个固定功能；成功结果统一输出 JSON，实时进度输出 JSON Lines。

## 项目目录

```text
Brain/
├── README.md                         # 英文文档
├── README_zh-CN.md                   # 简体中文文档
├── pyproject.toml                    # 包信息、依赖和 CLI 入口
├── uv.lock                           # 依赖锁文件
├── src/
│   └── brain/
│       ├── __init__.py
│       ├── cli/
│       │   ├── __init__.py
│       │   ├── ingest.py             # brain-ingest 入口
│       │   ├── search.py             # brain-search 入口
│       │   └── status.py             # brain-status 入口
│       ├── documents/
│       │   ├── __init__.py
│       │   ├── loaders.py            # 文件加载与 MinerU 解析
│       │   ├── cleaning.py           # 文本清洗与规范化
│       │   └── chunking.py           # 基于 token 预算的切片
│       ├── progress/
│       │   ├── __init__.py
│       │   ├── file_store.py         # progress.json 原子存储与心跳
│       │   └── models.py             # 入库任务模型
│       ├── storage/
│       │   ├── __init__.py
│       │   ├── client.py             # Elasticsearch 连接工具
│       │   └── elasticsearch_store.py # 增量发布与检索
│       ├── config.py                 # 基于环境变量的服务配置
│       ├── constants.py              # 支持的文件扩展名
│       ├── embeddings.py             # OpenAI 兼容与 Ollama 向量化
│       ├── ingestion.py              # 增量入库流程编排
│       ├── manifest.py               # 项目清单生成与恢复
│       ├── models.py                 # 文档、切片和召回模型
│       ├── project.py                # project 校验、锁和原子 JSON
│       ├── retrieval.py              # 多路召回服务
│       ├── runtime.py                # 运行时依赖构建
│       └── utils.py                  # 通用工具函数
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

## 配置

服务连接和处理参数从 `.env` 读取：

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

`input-dir`、`output-dir` 和 `project` 不从环境变量补全，必须通过对应 CLI 显式传入。

## 功能一：增量入库

```bash
uv run brain-ingest \
  --input-dir ./docs \
  --output-dir ./mvp_output \
  --project my-knowledge-base
```

三个参数全部必填。文件身份使用不区分大小写的 basename：

- 新文件名追加到 project。
- 同名文件替换该文件已有的全部 chunks。
- SHA-256 未变化的同名文件跳过。
- 本次输入未出现的历史文件继续保留。
- 同一批次出现重复 basename 时整批失败。
- 同一主机上的同一 project 同时只允许一个入库任务，即使传入不同的 output 目录也不能绕过锁。

入库采用 staging 索引和 Elasticsearch alias 原子切换。标准输出只包含最终 JSON，运行日志写入标准错误。
切换成功后按照 `ES_INDEX_VERSIONS_TO_KEEP` 回收旧物理索引；默认保留当前版本和 1 个回滚版本。

跳过未变化文件之前，会校验 manifest 记录的物理索引是否仍由当前 alias 指向，并核对 chunk 数。需要修复时，本次输入必须包含全部历史文件，系统才会自动重建，避免历史数据静默丢失。

项目产物目录：

```text
mvp_output/
└── my-knowledge-base/
    ├── manifest.json                 # project 当前文件清单
    ├── progress.json                 # 当前或最近一次入库状态
    ├── .ingest.lock                  # project 独占入库锁
    └── mineru/
        └── guide_<hash>/
            └── mineru_result.md      # MinerU 中间 Markdown
```

`manifest.json` 描述当前 Elasticsearch alias 中存储的资料，包括规则生成的 project 简介、主题、文件哈希、解析器、页数和 chunk 数。简介生成不调用 LLM。

## 功能二：列出全部 project

```bash
uv run brain-status --output-dir ./mvp_output
```

该命令扫描 output 目录下所有合法的 `manifest.json`，以 JSON 返回 project 简介和完整文件清单。损坏的 manifest 会作为带 `error` 的条目返回，不影响其他 project。

## 功能三：实时查看入库进度

```bash
uv run brain-status \
  --output-dir ./mvp_output \
  --project my-knowledge-base
```

任务运行时，该命令每秒读取一次 `progress.json`，只在状态变化时输出一行 JSON，直到任务成功、失败，或超过 30 秒没有心跳而进入 stale 状态。

阶段包括：

```text
recovering → scanning → parsing → cleaning → chunking
→ embedding → indexing → publishing → completed
```

## 功能四：指定 project 多路召回

```bash
uv run brain-search \
  --question "如何配置访问权限？" \
  --project my-knowledge-base \
  --top-k 10
```

三个参数全部必填，输出始终为 JSON。检索同时执行向量召回与关键词召回，再使用 RRF 融合：

- `top-k` 是最终返回数量，范围为 1–100。
- 每路候选数为 `max(2 × top-k, 20)`。
- 结果包含文件名、来源路径、页码、章节、原始正文、召回方法和 RRF 分数。
- project 不存在时返回 `project_not_found`；单路召回失败时成功结果包含 `warnings`，两路全部失败时返回 `retrieval_failed` 和非零退出码。
- 只召回原始知识块，不执行 LLM 答案生成。

## 支持文件

```text
PDF、DOCX、TXT、TEXT、Markdown、CSV、XLSX
```

配置 MinerU 时，PDF 优先使用 MinerU 解析；失败时回退到 pypdf。MinerU Markdown 保存在对应 project 的 `mineru/` 目录。

## 升级说明

现有 `docs_<workspace>_current` alias 会在首次增量运行时复制到新的 staging 版本。旧的 Elasticsearch 进度索引不再使用，旧 `mvp_output/mineru_*` 目录也不会自动移动。

## 测试

```bash
uv run pytest -q
```

单元测试使用隔离的 fake store。如需额外验证真实 Elasticsearch 的 bulk、alias、召回和旧版本回收链路：

```bash
BRAIN_TEST_ES_URL=http://localhost:9200 uv run pytest -q -m integration
```
