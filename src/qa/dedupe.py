from __future__ import annotations

import re

from src.models import QAEvidence, QAPair


def _normalize_question(q: str) -> str:
    n = re.sub(r"\s+", "", q)
    n = re.sub(r"[？?！!。,，：:；;、]", "", n)
    return n.lower()


def dedupe_qa(qa_pairs: list[QAPair]) -> list[QAPair]:
    merged: dict[str, QAPair] = {}
    for pair in qa_pairs:
        key = _normalize_question(pair.question)
        if key not in merged:
            merged[key] = pair
            continue
        existing = merged[key]
        ev_dict: dict[tuple[str, str], QAEvidence] = {}
        for e in existing.evidence + pair.evidence:
            k = (e.chunk_id, e.file_name)
            if k not in ev_dict or e.score > ev_dict[k].score:
                ev_dict[k] = e
        risk = sorted(set(existing.risk_notes + pair.risk_notes))
        ans = existing.answer if len(existing.answer) >= len(pair.answer) else pair.answer
        merged[key] = QAPair(
            question=existing.question,
            answer=ans,
            category=existing.category,
            risk_notes=risk,
            evidence=list(ev_dict.values()),
        )
    return list(merged.values())
