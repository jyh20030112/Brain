from src.models import QAEvidence, QAPair
from src.qa.dedupe import _normalize_question, dedupe_qa


def test_normalize_question_removes_spaces_and_common_punctuation():
    assert _normalize_question(" 这个 好用吗？ ") == "这个好用吗"
    assert _normalize_question("ABC，怎么样！") == "abc怎么样"


def test_dedupe_qa_merges_evidence_risk_notes_and_keeps_longer_answer():
    first = QAPair(
        question="这个好用吗？",
        answer="短回答",
        category="功效",
        risk_notes=["风险A"],
        evidence=[QAEvidence("c1", "a.md", "旧证据", 1, 0.2)],
    )
    second = QAPair(
        question="这个好用吗",
        answer="这是一个更完整的回答",
        category="功效",
        risk_notes=["风险B", "风险A"],
        evidence=[
            QAEvidence("c1", "a.md", "新证据", 1, 0.9),
            QAEvidence("c2", "b.md", "其他证据", 2, 0.5),
        ],
    )

    merged = dedupe_qa([first, second])

    assert len(merged) == 1
    assert merged[0].question == "这个好用吗？"
    assert merged[0].answer == "这是一个更完整的回答"
    assert merged[0].risk_notes == ["风险A", "风险B"]
    assert len(merged[0].evidence) == 2
    assert next(e for e in merged[0].evidence if e.chunk_id == "c1").score == 0.9
