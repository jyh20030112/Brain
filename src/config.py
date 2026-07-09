from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    # ---- 输入输出 ----
    input_dir: str = "./docs/"
    project: str = "省心说客服"
    output_dir: str = "mvp_output"

    # ---- LLM ----
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""

    # ---- Embedding ----
    embedding_provider: str = "openai"
    embedding_url: str = ""
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

    # ---- QA 生成参数 ----
    qa_limit: int = 12
    qa_generalization: int = 2
    chunk_size: int = 512
    chunk_overlap: int = 120

    @classmethod
    def from_env(cls) -> Config:
        """从环境变量加载配置（.env 文件需提前 load_dotenv）。"""
        import os

        return cls(
            input_dir=os.getenv("INPUT_DIR", "./docs/"),
            project=os.getenv("PROJECT", "省心说客服"),
            output_dir=os.getenv("OUTPUT_DIR", "mvp_output"),
            llm_base_url=os.getenv("LLM_BASE_URL", ""),
            llm_api_key=os.getenv("LLM_API_KEY", ""),
            llm_model=os.getenv("LLM_MODEL", ""),
            embedding_provider=os.getenv("EMBEDDING_PROVIDER", "openai"),
            embedding_url=os.getenv("EMBEDDING_URL", ""),
            embedding_model=os.getenv("EMBEDDING_MODEL", "bge-m3"),
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "1024")),
            es_cloud_id=os.getenv("ES_CLOUD_ID", ""),
            es_url=os.getenv("ES_URL", ""),
            es_username=os.getenv("ES_USERNAME", ""),
            es_password=os.getenv("ES_PASSWORD", ""),
            es_api_key=os.getenv("ES_API_KEY", ""),
            mineru_api_token=os.getenv("MINERU_API_TOKEN", ""),
            qa_limit=int(os.getenv("QA_LIMIT", "12")),
            qa_generalization=int(os.getenv("QA_GENERALIZATION", "2")),
            chunk_size=int(os.getenv("CHUNK_SIZE", "512")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "120")),
        )
