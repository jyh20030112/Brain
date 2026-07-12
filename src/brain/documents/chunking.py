from __future__ import annotations

import re

from brain.models import DocumentRecord, TextChunk
from brain.utils import _short_hash


_HEADING_PATTERN = re.compile(r"^[一二三四五六七八九十\d]+[、.)）]?\s*|^第\s*\d+\s*[步章节项]")
_TOKEN_PATTERN = re.compile(r"[\u3400-\u9fff]|[A-Za-z0-9_]+|[^\s]")
_SPLIT_BOUNDARY = re.compile(r"[\s。！？!?；;，,、：:]")


def _looks_like_heading(text: str) -> bool:
    """判断一行是否像章节标题（用于标记后续内容归属，不作为正文入块）"""
    if len(text) > 42:
        return False
    if _HEADING_PATTERN.search(text):
        return True
    return text.endswith(("：", ":")) and len(text) <= 30


def _estimate_token_count(text: str) -> int:
    """轻量估算 token 数：CJK/标点按 1，ASCII 连续词约每 4 字符 1 token。"""
    count = 0
    for match in _TOKEN_PATTERN.finditer(text):
        token = match.group(0)
        count += max(1, (len(token) + 3) // 4) if token.isascii() and token.replace("_", "").isalnum() else 1
    return count


def _max_prefix_index(text: str, budget: int) -> int:
    """返回不超过 token 预算的最大字符前缀位置。"""
    low, high = 1, len(text)
    best = 0
    while low <= high:
        middle = (low + high) // 2
        if _estimate_token_count(text[:middle]) <= budget:
            best = middle
            low = middle + 1
        else:
            high = middle - 1
    return max(1, best)


def _tail_with_budget(text: str, budget: int) -> str:
    if budget <= 0 or not text:
        return ""
    low, high = 0, len(text)
    best = len(text)
    while low <= high:
        middle = (low + high) // 2
        if _estimate_token_count(text[middle:]) <= budget:
            best = middle
            high = middle - 1
        else:
            low = middle + 1
    return text[best:].strip()


def _split_large(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """按估算 token 预算拆分，优先边界切分，并保证每块存在硬上限。"""
    remaining = text.strip()
    pieces: list[str] = []
    while remaining and _estimate_token_count(remaining) > chunk_size:
        hard_end = _max_prefix_index(remaining, chunk_size)
        prefix = remaining[:hard_end]
        boundaries = [match.end() for match in _SPLIT_BOUNDARY.finditer(prefix)]
        soft_end = boundaries[-1] if boundaries and boundaries[-1] >= hard_end // 2 else hard_end
        piece = remaining[:soft_end].strip()
        if not piece:
            soft_end = hard_end
            piece = remaining[:soft_end].strip()
        pieces.append(piece)
        overlap = _tail_with_budget(piece, chunk_overlap)
        remaining = f"{overlap}{remaining[soft_end:]}".strip()
    if remaining:
        pieces.append(remaining)
    return pieces


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
        metadata={**doc.metadata, "document_title": doc.title},
    )


def chunk_docs(
    documents: list[DocumentRecord],
    *,
    workspace_id: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[TextChunk]:
    """文档切块主流程：段落聚合 + token 预算拆分 + 带重叠滑窗 + 文档内去重。"""
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap 必须满足 0 <= overlap < chunk_size")

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
                    if buf:
                        chunks.append(_build_chunk(doc, workspace_id, page.page_number, section, buf))
                        buf = ""
                    section = p[:120]
                    continue
                candidate = p if not buf else f"{buf}\n{p}"
                if _estimate_token_count(candidate) <= chunk_size:
                    buf = candidate
                else:
                    if buf:
                        chunks.append(_build_chunk(doc, workspace_id, page.page_number, section, buf))
                        overlap = _tail_with_budget(buf, chunk_overlap)
                        candidate = f"{overlap}\n{p}".strip() if overlap else p
                    pieces = _split_large(candidate, chunk_size, chunk_overlap)
                    for piece in pieces[:-1]:
                        chunks.append(_build_chunk(doc, workspace_id, page.page_number, section, piece))
                    buf = pieces[-1] if pieces else ""
            if buf:
                chunks.append(_build_chunk(doc, workspace_id, page.page_number, section, buf))
    seen_ids, deduped = set(), []
    for c in chunks:
        if c.id in seen_ids:
            continue
        seen_ids.add(c.id)
        deduped.append(c)
    return deduped
