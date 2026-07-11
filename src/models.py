from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class DocumentPage:
    page_number: int | None
    text: str


@dataclass(slots=True)
class DocumentRecord:
    source_path: str
    file_name: str
    file_type: str
    title: str
    pages: list[DocumentPage]
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def raw_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text.strip())


@dataclass(slots=True)
class TextChunk:
    id: str
    workspace_id: str
    file_name: str
    source_path: str
    content: str
    page_number: int | None
    section: str
    chunk_type: str
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def embedding_text(self) -> str:
        """为向量化补充原始文档标题和章节语境，不改变入库正文。"""
        title = self.metadata.get("document_title", "").strip()
        rows = []
        if title:
            rows.append(f"文档标题：{title}")
        if self.section and self.section != title:
            rows.append(f"章节标题：{self.section}")
        rows.append(f"正文：{self.content}")
        return "\n".join(rows)


@dataclass(slots=True)
class RetrievedChunk:
    chunk: TextChunk
    score: float
    retrieval_method: str
