from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from src.llm.prompts import PROMPT_CUSTOMER_QUESTIONS, PROMPT_PRODUCT_DESC, PROMPT_QA_EXTRACT
from src.models import RetrievedChunk
from src.utils import _safe_excerpt


_GLM_THINKING_OFF = {"thinking": {"type": "disabled"}}
_QWEN_THINKING_OFF = {"enable_thinking": False}


class LLMClient:
    """LLM + Embedding 客户端，支持 OpenAI 兼容接口和 Ollama。"""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        embedding_url: str = "",
        embedding_model: str = "",
        embedding_dim: int = 1024,
        embedding_provider: str = "openai",
    ):
        from openai import OpenAI

        self.model = model
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self.embedding_provider = embedding_provider
        self._embedding_url = embedding_url
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=180.0, max_retries=1)
        if embedding_provider == "openai":
            emb_url = embedding_url or base_url
            self._emb_client: Any = OpenAI(api_key=api_key, base_url=emb_url, timeout=180.0, max_retries=0)
        else:
            self._emb_client = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts or not self.embedding_model:
            return []
        if self.embedding_provider == "ollama":
            return self._embed_ollama(texts)
        return self._embed_openai(texts)

    def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        batch_size = 64
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            for attempt in range(3):
                try:
                    resp = self._emb_client.embeddings.create(model=self.embedding_model, input=batch)
                    embeddings.extend([d.embedding for d in resp.data])
                    break
                except Exception:
                    if attempt == 2:
                        print(f"  [警告] embedding 第{i // batch_size + 1}批失败，已重试3次")
                        embeddings.extend([[] for _ in batch])
                    else:
                        time.sleep(min(2.0 * (attempt + 1), 5.0))
        return embeddings

    def _embed_ollama(self, texts: list[str]) -> list[list[float]]:
        """调用 Ollama 原生 /api/embed 接口，支持批量。"""
        import json as _json
        import urllib.request

        url = (self._embedding_url or "http://localhost:11434").rstrip("/") + "/api/embed"
        embeddings: list[list[float]] = []
        batch_size = 64
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            for attempt in range(3):
                try:
                    body = _json.dumps({"model": self.embedding_model, "input": batch}).encode("utf-8")
                    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(req, timeout=120) as resp:
                        data = _json.loads(resp.read().decode("utf-8"))
                        embeddings.extend(data.get("embeddings", [[] for _ in batch]))
                    break
                except Exception:
                    if attempt == 2:
                        print(f"  [警告] Ollama embedding 第{i // batch_size + 1}批失败，已重试3次")
                        embeddings.extend([[] for _ in batch])
                    else:
                        time.sleep(min(2.0 * (attempt + 1), 5.0))
        return embeddings

    def _thinking_off(self) -> dict:
        m = self.model.lower()
        return _GLM_THINKING_OFF if ("glm" in m or "kimi" in m) else _QWEN_THINKING_OFF

    def _extract_text(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                t = getattr(item, "text", None) if not isinstance(item, dict) else item.get("text")
                if isinstance(t, str) and t.strip():
                    parts.append(t.strip())
            return "\n".join(parts).strip()
        return str(content).strip()

    def chat(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> str:
        kwargs: dict = dict(
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            extra_body=self._thinking_off(),
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = self._client.chat.completions.create(**kwargs)
            return self._extract_text(resp.choices[0].message.content if resp.choices else "")
        except Exception as e:
            print(f"  [LLM错误] {e}")
            return ""

    def generate_product_description(self, context: str) -> str:
        return self.chat(
            "你是一个专业的产品文档撰写助手。",
            PROMPT_PRODUCT_DESC.format(context=context),
            temperature=0.3,
        )

    def _try_generate_questions(self, prompt: str) -> list[dict]:
        raw = self.chat("你是一个善于提问的潜在客户。请输出JSON格式。", prompt, temperature=0.7, json_mode=True)
        if not raw:
            return []
        try:
            data = json.loads(raw.replace("```json", "").replace("```", "").strip())
            if isinstance(data, list):
                return [i for i in data if isinstance(i, dict)]
            if isinstance(data, dict) and "questions" in data:
                qs = data["questions"]
                return [i for i in qs if isinstance(i, dict)] if isinstance(qs, list) else []
        except json.JSONDecodeError:
            pass
        return []

    def generate_customer_questions(
        self,
        product_description: str,
        count: int,
        generalization_count: int,
    ) -> list[dict]:
        prompt = PROMPT_CUSTOMER_QUESTIONS.format(
            product_description=product_description,
            count=count,
            generalization_count=generalization_count,
        )
        best: list[dict] = []
        for attempt in range(3):
            qs = self._try_generate_questions(prompt)
            n = sum(1 for q in qs if isinstance(q.get("question"), str) and q["question"].strip())
            if n >= count:
                return qs
            if len(qs) > len(best):
                best = qs
            if attempt < 2:
                time.sleep(5)
        return best

    def generate_answer(
        self,
        *,
        product_name: str,
        question: str,
        category: str,
        retrieved_chunks: list[RetrievedChunk],
    ) -> tuple[str, list[str]]:
        if not retrieved_chunks:
            return ("未在当前资料中检索到足够证据，建议客服谨慎回复并人工复核。", ["无检索证据"])
        context = _format_context(retrieved_chunks)
        prompt = PROMPT_QA_EXTRACT.format(
            product_name=product_name,
            category=category,
            question=question,
            context=context,
        )
        answer = self.chat("你是一个谨慎的客服知识库助手，只能根据提供证据回答。", prompt, temperature=0.2)
        if not answer:
            answer = _fallback_answer(question, retrieved_chunks)
        risk = _risk_notes(retrieved_chunks)
        return answer, risk

    def generate_answers_batch(
        self,
        *,
        product_name: str,
        questions: list[tuple[str, str]],
        retrieved_list: list[list[RetrievedChunk]],
    ) -> list[tuple[str, list[str]]]:
        if not questions:
            return []
        results: list = [("", [])] * len(questions)
        with ThreadPoolExecutor(max_workers=min(len(questions), 36)) as ex:
            futs = {
                ex.submit(
                    self.generate_answer,
                    product_name=product_name,
                    question=q[1],
                    category=q[0],
                    retrieved_chunks=retrieved,
                ): i
                for i, (q, retrieved) in enumerate(zip(questions, retrieved_list))
            }
            for fut in as_completed(futs):
                idx = futs[fut]
                try:
                    results[idx] = fut.result()
                except Exception:
                    results[idx] = (
                        _fallback_answer(questions[idx][1], retrieved_list[idx]),
                        _risk_notes(retrieved_list[idx]),
                    )
        return results


def _fallback_answer(question: str, retrieved_chunks: list[RetrievedChunk]) -> str:
    combined = " ".join(c.chunk.content for c in retrieved_chunks[:2])
    sentences = [s.strip() for s in re.split(r"(?<=[。！？!?；;])", combined) if s.strip()]
    if not sentences:
        return _safe_excerpt(combined, 220)
    selected = []
    for s in sentences:
        if len("".join(selected)) + len(s) > 180:
            break
        selected.append(s)
    ans = "".join(selected).strip()
    prefix = "根据资料，"
    if question and ans and not ans.startswith(prefix):
        return prefix + ans
    return ans or _safe_excerpt(combined, 220)


def _risk_notes(retrieved_chunks: list[RetrievedChunk]) -> list[str]:
    combined = " ".join(c.chunk.content for c in retrieved_chunks[:3])
    notes: list[str] = []
    if "不代表产品功效" in combined:
        notes.append('资料中包含"成分介绍不代表产品功效"的限制表述，回复时避免夸大宣传。')
    if any(kw in combined for kw in ["改善", "抗皱", "修护"]):
        notes.append("资料中涉及功效类表述，回复时注意措辞合规。")
    return notes


def _format_context(chunks: list[RetrievedChunk]) -> str:
    rows = []
    for i, rc in enumerate(chunks, start=1):
        page = f"第{rc.chunk.page_number}页" if rc.chunk.page_number is not None else "页码未知"
        rows.append(
            f"[证据{i}] 文件={rc.chunk.file_name} | {page} | 分数={rc.score:.3f}\n"
            f"{_safe_excerpt(rc.chunk.content, 400)}"
        )
    return "\n\n".join(rows)
