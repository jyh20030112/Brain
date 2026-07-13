from simbrain.documents.chunking import _estimate_token_count, _looks_like_heading, chunk_docs
from simbrain.models import DocumentPage, DocumentRecord


def _doc(text: str) -> DocumentRecord:
    return DocumentRecord(
        source_path="source.md",
        file_name="source.md",
        file_type="markdown",
        title="默认标题",
        pages=[DocumentPage(page_number=1, text=text)],
    )


def test_looks_like_heading_recognizes_numbered_and_colon_headings():
    assert _looks_like_heading("一、使用方法")
    assert _looks_like_heading("注意事项：")
    assert not _looks_like_heading("这是一段比较长的正文内容，不应该被当成标题处理，因为它承载了完整语义。")


def test_chunk_docs_uses_heading_as_section_and_marks_tables():
    chunks = chunk_docs(
        [_doc("一、使用方法\n\n早晚洁面后使用。\n\ncolumns: 名称 | 容量\nrow 1: 精华=30ml")],
        workspace_id="wid",
        chunk_size=80,
        chunk_overlap=10,
    )

    assert len(chunks) == 1
    assert chunks[0].section == "一、使用方法"
    assert chunks[0].chunk_type == "table"
    assert "早晚洁面后使用" in chunks[0].content


def test_chunk_docs_splits_large_paragraphs_and_preserves_cross_document_sources():
    text = "第一句内容比较长。第二句内容也比较长。第三句内容还是比较长。"
    duplicate = DocumentRecord(
        source_path="duplicate.md",
        file_name="duplicate.md",
        file_type="markdown",
        title="默认标题",
        pages=[DocumentPage(page_number=1, text="重复内容。")],
    )

    chunks = chunk_docs([_doc(text), _doc("重复内容。"), duplicate], workspace_id="wid", chunk_size=18, chunk_overlap=4)

    assert len(chunks) >= 3
    duplicate_chunks = [c for c in chunks if c.content == "重复内容。"]
    assert {c.source_path for c in duplicate_chunks} == {"source.md", "duplicate.md"}
    assert all(_estimate_token_count(c.content) <= 18 for c in chunks)
    assert all(c.id.startswith("chunk_") for c in chunks)


def test_chunk_docs_hard_splits_unpunctuated_text_within_token_budget():
    chunks = chunk_docs([_doc("A" * 160)], workspace_id="wid", chunk_size=10, chunk_overlap=2)

    assert len(chunks) > 1
    assert all(_estimate_token_count(c.content) <= 10 for c in chunks)


def test_chunk_docs_flushes_content_before_switching_section():
    chunks = chunk_docs(
        [_doc("前一章节正文。\n\n二、后一章节\n\n后一章节正文。")],
        workspace_id="wid",
        chunk_size=30,
        chunk_overlap=3,
    )

    assert chunks[0].section == "默认标题"
    assert chunks[1].section == "二、后一章节"
