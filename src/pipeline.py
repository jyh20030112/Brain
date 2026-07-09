from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from src.config import Config
from src.constants import SUPPORTED_EXTENSIONS
from src.documents.chunking import chunk_docs
from src.documents.cleaning import clean_docs
from src.documents.loaders import load_docs
from src.llm.client import LLMClient
from src.llm.prompts import DEFAULT_QUESTIONS
from src.models import DocumentPage, DocumentRecord
from src.output.exporters import qa_pairs_to_markdown, qa_pairs_to_rows
from src.qa.dedupe import dedupe_qa
from src.qa.generation import generate_qa_pairs
from src.storage.elasticsearch_store import ESStore


def run_pipeline(cfg: Config) -> None:
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
    for fp in file_paths:
        print(f"         - {fp.name}")

    output_dir = Path(cfg.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[输出]   {output_dir}")

    workspace_id = hashlib.md5(cfg.project.encode()).hexdigest()[:16]
    print(f"[ID]     {workspace_id}")

    if cfg.embedding_provider == "ollama":
        embedding_url = cfg.embedding_url or "http://localhost:11434"
    else:
        embedding_url = cfg.embedding_url or cfg.llm_base_url
    print(f"[LLM]    {cfg.llm_model} @ {cfg.llm_base_url}")
    print(f"[Embed]  {cfg.embedding_model} ({cfg.embedding_provider}) @ {embedding_url}")
    print(f"[ES]     {cfg.es_url}")

    llm = LLMClient(
        base_url=cfg.llm_base_url,
        api_key=cfg.llm_api_key,
        model=cfg.llm_model,
        embedding_url=embedding_url,
        embedding_model=cfg.embedding_model,
        embedding_dim=cfg.embedding_dim,
        embedding_provider=cfg.embedding_provider,
    )
    es = ESStore(
        workspace_id=workspace_id,
        es_url=cfg.es_url,
        es_cloud_id=cfg.es_cloud_id,
        es_user=cfg.es_username,
        es_pass=cfg.es_password,
        es_api_key=cfg.es_api_key,
        embedding_dim=cfg.embedding_dim,
    )

    print("\n── 功能1: 文档加载 + 清洗 ──")
    print(f"  MinerU: {'禁用' if not cfg.mineru_api_token else '已配置（云 API）'}")
    print("  加载文档...")
    docs = load_docs(file_paths, mineru_api_token=cfg.mineru_api_token, output_dir=output_dir)
    if not docs:
        print("错误: 没有可处理的文档")
        raise SystemExit(1)
    print(f"  已加载 {len(docs)} 个文档")

    print("  清洗文档...")
    docs = clean_docs(docs)

    print("\n── 产品描述生成 ──")
    all_text = "\n\n".join(d.raw_text for d in docs)
    desc = llm.generate_product_description(all_text[:100000])

    product_name = "未命名产品"
    for line in (desc or "").split("\n"):
        if line.strip().startswith("# "):
            product_name = line.strip().replace("# ", "").strip()
            break
    print(f"  产品名: {product_name}")

    docs.append(
        DocumentRecord(
            source_path="generated/product_description.md",
            file_name="product_description.md",
            file_type="markdown",
            title="产品介绍文档",
            pages=[DocumentPage(page_number=1, text=desc)],
            metadata={"type": "generated"},
        )
    )

    print(f"\n── 文档切块 (size={cfg.chunk_size}, overlap={cfg.chunk_overlap}) ──")
    chunks = chunk_docs(docs, workspace_id=workspace_id, chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap)
    print(f"  共 {len(chunks)} 个 chunk")
    if not chunks:
        print("错误: 切块结果为空")
        raise SystemExit(1)

    print("\n── 功能3: docs 写入 ES ──")
    chunk_texts = [c.content for c in chunks]
    print(f"  生成 embedding ({len(chunks)} 条)...")
    chunk_embs = llm.embed(chunk_texts)
    es.index_docs(chunks, chunk_embs)

    print("\n── 功能2: QA 生成 ──")
    print("  生成客户问题...")
    questions_data = llm.generate_customer_questions(
        desc,
        count=cfg.qa_limit,
        generalization_count=cfg.qa_generalization,
    )
    questions_list = []
    for item in questions_data:
        cat = item.get("category", "通用")
        q = item.get("question", "")
        if q:
            questions_list.append((cat, q))
        for v in item.get("variations", []):
            if v:
                questions_list.append((cat, v))
    if not questions_list:
        questions_list = [(c, q.format(name=product_name)) for c, q in DEFAULT_QUESTIONS]
    print(f"  共 {len(questions_list)} 个问题（含变体）")

    print("  批量检索 + 生成答案...")
    qa_pairs = generate_qa_pairs(
        product_name=product_name,
        llm=llm,
        es=es,
        questions=questions_list,
        batch_size=cfg.qa_limit + cfg.qa_limit * cfg.qa_generalization,
    )

    print("  去重...")
    qa_pairs = dedupe_qa(qa_pairs)
    print(f"  去重后 {len(qa_pairs)} 条 QA")

    print("\n── 功能3: QA 写入 ES ──")
    qa_texts = [f"{qa.question} {qa.answer}" for qa in qa_pairs]
    print(f"  生成 QA embedding ({len(qa_pairs)} 条)...")
    qa_embs = llm.embed(qa_texts)
    es.index_qa(qa_pairs, qa_embs)

    print("\n── 产物输出 ──")
    output_paths = {}

    jp = output_dir / "qa_list.json"
    jp.write_text(
        json.dumps(
            {
                "product_name": product_name,
                "qa_pairs": [p.to_dict() for p in qa_pairs],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    output_paths["json"] = str(jp)

    mp = output_dir / "qa_list.md"
    mp.write_text(qa_pairs_to_markdown(product_name, qa_pairs), encoding="utf-8")
    output_paths["markdown"] = str(mp)

    cp = output_dir / "qa_list.csv"
    with cp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["question", "answer", "category", "risk_notes", "evidence_files"])
        w.writeheader()
        w.writerows(qa_pairs_to_rows(qa_pairs))
    output_paths["csv"] = str(cp)

    dp = output_dir / "product_description.md"
    dp.write_text(desc, encoding="utf-8")
    output_paths["product_description"] = str(dp)

    for label, path in sorted(output_paths.items()):
        print(f"  {label}: {path}")

    print(f"\n{'=' * 60}")
    print("[完成]")
    print(f"  产品名:   {product_name}")
    print(f"  Chunk 数: {len(chunks)}")
    print(f"  QA 对数:  {len(qa_pairs)}")
    print(f"  输出目录: {output_dir}")
