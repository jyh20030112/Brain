from simbrain.config import Config


def test_config_defaults_match_env_example(monkeypatch):
    for name in (
        "EMBEDDING_PROVIDER",
        "EMBEDDING_URL",
        "EMBEDDING_MODEL",
        "ES_USERNAME",
    ):
        monkeypatch.delenv(name, raising=False)

    config = Config.from_env()

    assert config.embedding_provider == "ollama"
    assert config.embedding_url == "http://localhost:11434"
    assert config.embedding_model == "bge-m3"
    assert config.es_username == "elastic"
