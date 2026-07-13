from simbrain.documents.cleaning import clean_docs, clean_text
from simbrain.models import DocumentPage, DocumentRecord


def test_clean_text_normalizes_control_chars_whitespace_and_repeated_lines():
    text = "标题\x00\x01\r\n　hello\xa0  world\n重复\n重复\n-- 1 of 2 --\n中 文 标\n\n\n尾部"

    cleaned = clean_text(text)

    assert "\x00" not in cleaned
    assert "\x01" not in cleaned
    assert "\r" not in cleaned
    assert "hello world" in cleaned
    assert cleaned.count("重复") == 1
    assert "中文标" in cleaned
    assert "\n\n\n" not in cleaned


def test_clean_docs_returns_cleaned_document_records():
    doc = DocumentRecord(
        source_path="a.txt",
        file_name="a.txt",
        file_type="txt",
        title=" 标 题 ",
        pages=[DocumentPage(page_number=1, text="A\xa0  B")],
        metadata={"k": "v"},
    )

    result = clean_docs([doc])

    assert result[0].title == "标 题"
    assert result[0].pages[0].text == "A B"
    assert result[0].metadata == {"k": "v"}
