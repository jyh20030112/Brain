from __future__ import annotations

import re

from src.models import DocumentRecord, TextChunk
from src.utils import _short_hash


_HEADING_PATTERN = re.compile(r"^[一二三四五六七八九十\d]+[、.)）]?\s*|^第\s*\d+\s*[步章节项]")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？!?；;])")


def _looks_like_heading(text: str) -> bool:
    """判断一行是否像章节标题（用于标记后续内容归属，不作为正文入块）"""
    if len(text) > 42:
        return False
    if _HEADING_PATTERN.search(text):
        return True
    return text.endswith(("：", ":")) and len(text) <= 30


def _split_large(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """将超长段落按句子边界拆成多个块，块间保留 chunk_overlap 字符的重叠"""
    sentences = [s.strip() for s in _SENTENCE_BOUNDARY.split(text) if s.strip()]
    if not sentences:
        return [text[:chunk_size]]
    pieces, buf = [], ""
    for s in sentences:
        if not buf:
            buf = s
            continue
        cand = f"{buf}{s}"
        if len(cand) <= chunk_size:
            buf = cand
            continue
        pieces.append(buf)
        overlap = buf[-chunk_overlap:] if chunk_overlap > 0 else ""
        buf = f"{overlap}{s}"
    if buf:
        pieces.append(buf)
    return [p.strip() for p in pieces if p.strip()]


def _build_chunk(
    doc: DocumentRecord,
    workspace_id: str,
    page_number: int | None,
    section: str,
    content: str,
) -> TextChunk:
    """构造 TextChunk 对象：推断类型、生成幂等 id、附带章节与标题元数据"""
    ct = "table" if "columns:" in content or "row " in content else "paragraph"
    seed = f"{doc.source_path}:{page_number}:{section}:{content}"
    return TextChunk(
        id=f"chunk_{_short_hash(seed)}",
        workspace_id=workspace_id,
        file_name=doc.file_name,
        source_path=doc.source_path,
        content=content.strip(),
        page_number=page_number,
        section=section[:120],
        chunk_type=ct,
        metadata={"document_title": doc.title},
    )


def chunk_docs(
    documents: list[DocumentRecord],
    *,
    workspace_id: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[TextChunk]:
    """文档切块主流程：段落优先聚合 + 超长段句级拆分 + 带重叠滑窗 + 去重"""
    chunks: list[TextChunk] = []
    for doc in documents:
        section = doc.title or doc.file_name
        for page in doc.pages:
            paragraphs = [p.strip() for p in re.split(r"\n{2,}", page.text) if p.strip()]
            buf = ""
            for para in paragraphs:
                p = re.sub(r"\s+", " ", para).strip()
                if not p:
                    continue
                if _looks_like_heading(p):
                    section = p[:120]
                    continue
                if len(p) > chunk_size:
                    if buf:
                        chunks.append(_build_chunk(doc, workspace_id, page.page_number, section, buf))
                        buf = ""
                    for piece in _split_large(p, chunk_size, chunk_overlap):
                        chunks.append(_build_chunk(doc, workspace_id, page.page_number, section, piece))
                    continue
                if not buf:
                    buf = p
                    continue
                cand = f"{buf}\n{p}"
                if len(cand) <= chunk_size:
                    buf = cand
                else:
                    chunks.append(_build_chunk(doc, workspace_id, page.page_number, section, buf))
                    buf = f"{buf[-chunk_overlap:].strip()}\n{p}" if chunk_overlap > 0 else p
            if buf:
                chunks.append(_build_chunk(doc, workspace_id, page.page_number, section, buf))
    seen_content, seen_ids, deduped = set(), set(), []
    for c in chunks:
        sig = c.content.strip()
        if sig in seen_content or c.id in seen_ids:
            continue
        seen_content.add(sig)
        seen_ids.add(c.id)
        deduped.append(c)
    return deduped
