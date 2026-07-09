from src.models import QAEvidence, QAPair
from src.output.exporters import qa_pairs_to_markdown, qa_pairs_to_rows


def test_qa_pairs_to_markdown_includes_answer_risks_and_evidence():
    pair = QAPair(
        question="怎么用？",
        answer="早晚洁面后使用。",
        category="使用方法",
        risk_notes=["避免夸大"],
        evidence=[QAEvidence("c1", "manual.md", "早晚使用", 3, 0.8765)],
    )

    markdown = qa_pairs_to_markdown("产品A", [pair])

    assert markdown.startswith("# 产品A QA 列表")
    assert "## 1. 怎么用？" in markdown
    assert "风险提示：" in markdown
    assert "- 避免夸大" in markdown
    assert "manual.md / 第3页 / score=0.876 / 早晚使用" in markdown


def test_qa_pairs_to_rows_flattens_pairs_for_csv():
    pair = QAPair(
        question="怎么用？",
        answer="早晚洁面后使用。",
        category="使用方法",
        risk_notes=["避免夸大", "人工复核"],
        evidence=[
            QAEvidence("c1", "manual.md", "早晚使用", 3, 0.8),
            QAEvidence("c2", "manual.md", "重复文件", 4, 0.7),
            QAEvidence("c3", "faq.md", "补充说明", 5, 0.6),
        ],
    )

    rows = qa_pairs_to_rows([pair])

    assert rows == [
        {
            "question": "怎么用？",
            "answer": "早晚洁面后使用。",
            "category": "使用方法",
            "risk_notes": "避免夸大 | 人工复核",
            "evidence_files": "manual.md | faq.md",
        }
    ]
