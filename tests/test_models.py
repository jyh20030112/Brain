from brain.models import TextChunk


def test_embedding_text_adds_title_and_section_without_changing_content():
    chunk = TextChunk(
        id="chunk_1",
        workspace_id="wid",
        file_name="manual.md",
        source_path="manual.md",
        content="早晚洁面后使用。",
        page_number=1,
        section="使用方法",
        chunk_type="paragraph",
        metadata={"document_title": "产品手册"},
    )

    assert chunk.embedding_text == "文档标题：产品手册\n章节标题：使用方法\n正文：早晚洁面后使用。"
    assert chunk.content == "早晚洁面后使用。"
