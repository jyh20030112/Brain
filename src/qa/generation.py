from __future__ import annotations

from src.llm.client import LLMClient
from src.models import QAEvidence, QAPair, RetrievedChunk
from src.storage.elasticsearch_store import ESStore
from src.utils import _safe_excerpt


def generate_qa_pairs(
    *,
    product_name: str,
    llm: LLMClient,
    es: ESStore,
    questions: list[tuple[str, str]],
    top_k: int = 5,
    batch_size: int = 36,
) -> list[QAPair]:
    qa_pairs: list[QAPair] = []
    total = len(questions)
    batch_size = max(1, batch_size)

    for bstart in range(0, total, batch_size):
        bend = min(bstart + batch_size, total)
        batch = questions[bstart:bend]
        print(f"  [QA] 批次 [{bstart + 1}-{bend}]/{total}")

        retrieved_list: list[list[RetrievedChunk]] = []
        for idx, (category, question) in enumerate(batch, start=bstart + 1):
            print(f"  [QA] [{idx}] {question}")
            q_emb = llm.embed([question])
            q_vec = q_emb[0] if q_emb else []
            retrieved = es.search_docs(question, q_vec, top_k=top_k)
            retrieved_list.append(retrieved)

        results = llm.generate_answers_batch(
            product_name=product_name,
            questions=batch,
            retrieved_list=retrieved_list,
        )

        for (category, question), retrieved, (answer, risk_notes) in zip(batch, retrieved_list, results):
            evidence = [
                QAEvidence(
                    chunk_id=rc.chunk.id,
                    file_name=rc.chunk.file_name,
                    excerpt=_safe_excerpt(rc.chunk.content),
                    page_number=rc.chunk.page_number,
                    score=rc.score,
                )
                for rc in retrieved[:top_k]
            ]
            qa_pairs.append(
                QAPair(
                    question=question,
                    answer=answer,
                    category=category,
                    risk_notes=risk_notes,
                    evidence=evidence,
                )
            )

    return qa_pairs
