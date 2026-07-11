from __future__ import annotations

import sys
from pathlib import Path

from brain.config import Config
from brain.constants import SUPPORTED_EXTENSIONS
from brain.documents.chunking import chunk_docs
from brain.documents.cleaning import clean_docs
from brain.documents.loaders import load_docs
from brain.progress.store import ProgressStore
from brain.runtime import build_embedding_client, build_es_store, build_progress_store


def run_ingestion(cfg: Config) -> str:
    cfg.validate_for_ingestion()
    progress = build_progress_store(cfg)
    job = progress.create_job(project=cfg.project, workspace_id=cfg.workspace_id)
    print(f"[任务]   {job.job_id}")
    try:
        _execute_ingestion(cfg, progress, job.job_id)
    except BaseException as exc:
        try:
            progress.fail_job(job.job_id, error=f"{type(exc).__name__}: {exc}")
        except Exception as progress_exc:
            print(f"[进度]   无法记录失败状态: {progress_exc}", file=sys.stderr)
        raise
    return job.job_id


def _execute_ingestion(cfg: Config, progress: ProgressStore, job_id: str) -> None:
    input_dir = Path(cfg.input_dir).resolve()
    if not input_dir.is_dir():
        print(f"错误: input_dir 路径不存在: {input_dir}")
        raise SystemExit(1)

    print(f"[输入]   扫描 {input_dir}")
    file_paths = [fp for fp in input_dir.rglob("*") if fp.is_file() and fp.suffix.lower() in SUPPORTED_EXTENSIONS]
    file_paths.sort(key=lambda p: p.name)
    if not file_paths:
        print("错误: 未找到受支持的文档文件")
        raise SystemExit(1)
    progress.update_job(
        job_id,
        stage="scanning",
        current=len(file_paths),
        total=len(file_paths),
        documents_total=len(file_paths),
    )
    for fp in file_paths:
        print(f"         - {fp.name}")

    output_dir = Path(cfg.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[输出]   {output_dir}")

    workspace_id = cfg.workspace_id
    print(f"[ID]     {workspace_id}")

    embedding_url = cfg.embedding_base_url
    print(f"[Embed]  {cfg.embedding_model} ({cfg.embedding_provider}) @ {embedding_url}")
    print(f"[ES]     {cfg.es_url}")

    embeddings = build_embedding_client(cfg)
    es = build_es_store(cfg)

    print("\n── 文档加载 + 清洗 ──")
    print(f"  MinerU: {'禁用' if not cfg.mineru_api_token else '已配置（云 API）'}")
    print("  加载文档...")
    progress.update_job(job_id, stage="parsing", current=0, total=len(file_paths))
    docs = load_docs(
        file_paths,
        mineru_api_token=cfg.mineru_api_token,
        output_dir=output_dir,
        progress_callback=lambda current, total: progress.update_job(
            job_id,
            stage="parsing",
            current=current,
            total=total,
            documents_succeeded=current,
        ),
    )
    if not docs:
        print("错误: 没有可处理的文档")
        raise SystemExit(1)
    print(f"  已加载 {len(docs)} 个文档")

    print("  清洗文档...")
    progress.update_job(job_id, stage="cleaning", current=0, total=len(docs))
    docs = clean_docs(docs)
    progress.update_job(job_id, stage="cleaning", current=len(docs), total=len(docs))

    print(f"\n── 文档切块 (size={cfg.chunk_size}, overlap={cfg.chunk_overlap}) ──")
    progress.update_job(job_id, stage="chunking", current=0, total=len(docs))
    chunks = chunk_docs(docs, workspace_id=workspace_id, chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap)
    print(f"  共 {len(chunks)} 个 chunk")
    if not chunks:
        print("错误: 切块结果为空")
        raise SystemExit(1)
    progress.update_job(
        job_id,
        stage="chunking",
        current=len(docs),
        total=len(docs),
        chunks_total=len(chunks),
    )

    print("\n── 原始资料写入 ES ──")
    chunk_texts = [c.embedding_text for c in chunks]
    print(f"  生成 embedding ({len(chunks)} 条)...")
    progress.update_job(job_id, stage="embedding", current=0, total=len(chunks))
    chunk_embs = embeddings.embed(
        chunk_texts,
        progress_callback=lambda current, total: progress.update_job(
            job_id,
            stage="embedding",
            current=current,
            total=total,
        ),
    )
    progress.update_job(job_id, stage="indexing", current=0, total=len(chunks))
    active_index = es.index_docs(
        chunks,
        chunk_embs,
        progress_callback=lambda current, total: progress.update_job(
            job_id,
            stage="indexing",
            current=current,
            total=total,
        ),
        publishing_callback=lambda: progress.update_job(
            job_id,
            stage="publishing",
            current=0,
            total=1,
        ),
    )
    progress.complete_job(job_id, active_index=active_index)

    print(f"\n{'=' * 60}")
    print("[完成]")
    print(f"  文档数:   {len(docs)}")
    print(f"  Chunk 数: {len(chunks)}")
    print(f"  活跃索引: {active_index}")
