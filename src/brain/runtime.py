from __future__ import annotations

from brain.config import Config
from brain.embeddings import EmbeddingClient
from brain.storage.elasticsearch_store import ESStore


def build_embedding_client(cfg: Config) -> EmbeddingClient:
    return EmbeddingClient(
        provider=cfg.embedding_provider,
        base_url=cfg.embedding_base_url,
        api_key=cfg.embedding_api_key,
        model=cfg.embedding_model,
    )


def build_es_store(cfg: Config) -> ESStore:
    return ESStore(
        workspace_id=cfg.workspace_id,
        es_url=cfg.es_url,
        es_cloud_id=cfg.es_cloud_id,
        es_user=cfg.es_username,
        es_pass=cfg.es_password,
        es_api_key=cfg.es_api_key,
        embedding_dim=cfg.embedding_dim,
    )
