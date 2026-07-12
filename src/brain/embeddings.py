from __future__ import annotations

import json
import time
from typing import Any, Callable


class EmbeddingClient:
    """Embedding 客户端，支持 OpenAI 兼容接口和 Ollama。"""

    def __init__(
        self,
        *,
        provider: str,
        base_url: str,
        api_key: str = "",
        model: str,
    ):
        if provider not in {"openai", "ollama"}:
            raise ValueError(f"不支持的 embedding_provider: {provider}")
        if not model:
            raise ValueError("embedding_model 不能为空")

        self.provider = provider
        self.base_url = base_url.rstrip("/")
        self.model = model
        if provider == "openai":
            from openai import OpenAI

            self._client: Any = OpenAI(api_key=api_key, base_url=self.base_url, timeout=180.0, max_retries=2)
        else:
            self._client = None

    def embed(
        self,
        texts: list[str],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[list[float]]:
        if not texts:
            return []
        if self.provider == "ollama":
            return self._embed_ollama(texts, progress_callback)
        return self._embed_openai(texts, progress_callback)

    def _embed_openai(
        self,
        texts: list[str],
        progress_callback: Callable[[int, int], None] | None,
    ) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for index in range(0, len(texts), 64):
            batch = texts[index : index + 64]
            response = self._client.embeddings.create(model=self.model, input=batch)
            vectors = [item.embedding for item in response.data]
            if len(vectors) != len(batch):
                raise ValueError(f"Embedding API 返回 {len(vectors)} 条向量，预期 {len(batch)} 条")
            embeddings.extend(vectors)
            if progress_callback:
                progress_callback(len(embeddings), len(texts))
        return embeddings

    def _embed_ollama(
        self,
        texts: list[str],
        progress_callback: Callable[[int, int], None] | None,
    ) -> list[list[float]]:
        import urllib.request

        url = f"{self.base_url}/api/embed"
        embeddings: list[list[float]] = []
        for index in range(0, len(texts), 64):
            batch = texts[index : index + 64]
            for attempt in range(3):
                try:
                    body = json.dumps({"model": self.model, "input": batch}).encode("utf-8")
                    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(request, timeout=120) as response:
                        payload = json.loads(response.read().decode("utf-8"))
                    vectors = payload.get("embeddings")
                    if not isinstance(vectors, list) or len(vectors) != len(batch):
                        count = len(vectors) if isinstance(vectors, list) else 0
                        raise ValueError(f"Ollama 返回 {count} 条向量，预期 {len(batch)} 条")
                    break
                except Exception as exc:
                    if attempt == 2:
                        raise RuntimeError(f"Ollama embedding 第 {index // 64 + 1} 批连续 3 次失败") from exc
                    time.sleep(min(2.0 * (attempt + 1), 5.0))
            embeddings.extend(vectors)
            if progress_callback:
                progress_callback(len(embeddings), len(texts))
        return embeddings
