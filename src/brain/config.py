from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass
class Config:
    # ---- 输入输出 ----
    input_dir: str = "./docs/"
    project: str = "default-knowledge-base"
    output_dir: str = "mvp_output"

    # ---- Embedding ----
    embedding_provider: str = "openai"
    embedding_url: str = ""
    embedding_api_key: str = ""
    embedding_model: str = ""
    embedding_dim: int = 1024

    # ---- Elasticsearch (Cloud ID / URL + 认证) ----
    es_cloud_id: str = ""
    es_url: str = ""
    es_username: str = ""
    es_password: str = ""
    es_api_key: str = ""

    # ---- MinerU (PDF OCR，可选) ----
    mineru_api_token: str = ""

    # ---- 切片参数 ----
    chunk_size: int = 512
    chunk_overlap: int = 120

    @property
    def workspace_id(self) -> str:
        return hashlib.md5(self.project.encode()).hexdigest()[:16]

    @property
    def embedding_base_url(self) -> str:
        return self.embedding_url or "http://localhost:11434"

    def _common_errors(self) -> list[str]:
        errors: list[str] = []
        if not self.project.strip():
            errors.append("PROJECT 不能为空")
        if self.embedding_provider not in {"openai", "ollama"}:
            errors.append("EMBEDDING_PROVIDER 仅支持 openai 或 ollama")
        if not self.embedding_model:
            errors.append("EMBEDDING_MODEL 不能为空")
        if self.embedding_provider == "openai" and not self.embedding_url:
            errors.append("使用 OpenAI 兼容 embedding 时必须配置 EMBEDDING_URL")
        if self.embedding_dim <= 0:
            errors.append("EMBEDDING_DIM 必须大于 0")
        if not (self.es_cloud_id or self.es_url):
            errors.append("必须配置 ES_CLOUD_ID 或 ES_URL")
        return errors

    def validate_for_ingestion(self) -> None:
        errors = self._common_errors()
        if self.chunk_size <= 0:
            errors.append("CHUNK_SIZE 必须大于 0")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            errors.append("CHUNK_OVERLAP 必须满足 0 <= overlap < CHUNK_SIZE")
        if errors:
            raise ValueError("配置校验失败：" + "；".join(errors))

    def validate_for_search(self) -> None:
        errors = self._common_errors()
        if errors:
            raise ValueError("配置校验失败：" + "；".join(errors))

    @classmethod
    def from_env(cls) -> Config:
        """从环境变量加载配置（.env 文件需提前 load_dotenv）。"""
        import os

        return cls(
            input_dir=os.getenv("INPUT_DIR", "./docs/"),
            project=os.getenv("PROJECT", "default-knowledge-base"),
            output_dir=os.getenv("OUTPUT_DIR", "mvp_output"),
            embedding_provider=os.getenv("EMBEDDING_PROVIDER", "openai"),
            embedding_url=os.getenv("EMBEDDING_URL", ""),
            # 兼容原先复用 LLM_API_KEY 的部署；新配置应使用 EMBEDDING_API_KEY。
            embedding_api_key=os.getenv("EMBEDDING_API_KEY", os.getenv("LLM_API_KEY", "")),
            embedding_model=os.getenv("EMBEDDING_MODEL", "bge-m3"),
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "1024")),
            es_cloud_id=os.getenv("ES_CLOUD_ID", ""),
            es_url=os.getenv("ES_URL", ""),
            es_username=os.getenv("ES_USERNAME", ""),
            es_password=os.getenv("ES_PASSWORD", ""),
            es_api_key=os.getenv("ES_API_KEY", ""),
            mineru_api_token=os.getenv("MINERU_API_TOKEN", ""),
            chunk_size=int(os.getenv("CHUNK_SIZE", "512")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "120")),
        )
