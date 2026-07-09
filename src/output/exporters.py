from __future__ import annotations

from src.models import QAPair


def qa_pairs_to_markdown(product_name: str, qa_pairs: list[QAPair]) -> str:
    lines = [f"# {product_name} QA 列表", ""]
    for i, pair in enumerate(qa_pairs, start=1):
        lines.append(f"## {i}. {pair.question}")
        lines.append(pair.answer)
        if pair.risk_notes:
            lines.append("")
            lines.append("风险提示：")
            for n in pair.risk_notes:
                lines.append(f"- {n}")
        if pair.evidence:
            lines.append("")
            lines.append("证据来源：")
            for e in pair.evidence:
                page = f"第{e.page_number}页" if e.page_number is not None else "页码未知"
                lines.append(f"- {e.file_name} / {page} / score={e.score:.3f} / {e.excerpt}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def qa_pairs_to_rows(qa_pairs: list[QAPair]) -> list[dict[str, str]]:
    return [
        {
            "question": p.question,
            "answer": p.answer,
            "category": p.category,
            "risk_notes": " | ".join(p.risk_notes),
            "evidence_files": " | ".join(dict.fromkeys(e.file_name for e in p.evidence)),
        }
        for p in qa_pairs
    ]
