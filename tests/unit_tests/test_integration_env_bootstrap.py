import os

from tests.integration_tests.env_bootstrap import configure_integration_test_env


def test_configure_integration_test_env_rewrites_datastores_to_localhost_on_host(monkeypatch):
    monkeypatch.setenv("POSTGRES_HOST", "prokaryotes-postgres")
    monkeypatch.setenv("POSTGRES_USER", "someone")
    monkeypatch.setenv("POSTGRES_PASSWORD", "secret")
    monkeypatch.setenv("POSTGRES_DB", "otherdb")
    monkeypatch.setenv("REDIS_HOST", "prokaryotes-redis")
    monkeypatch.setenv("ELASTIC_URI", "http://prokaryotes-elasticsearch:9200")

    configure_integration_test_env(running_in_docker=False)

    assert os.environ["POSTGRES_HOST"] == "localhost"
    assert os.environ["POSTGRES_USER"] == "postgres"
    assert os.environ["POSTGRES_PASSWORD"] == "Ma9icMicr0be"
    assert os.environ["POSTGRES_DB"] == "prokaryotes"
    assert os.environ["REDIS_HOST"] == "localhost"
    assert os.environ["ELASTIC_URI"] == "http://localhost:9200"
    assert os.environ["COMPACTION_TOKEN_THRESHOLD_PCT"] == "1"
    assert os.environ["COMPACTION_RECENCY_TAIL"] == "2"


def test_configure_integration_test_env_preserves_container_env(monkeypatch):
    monkeypatch.setenv("POSTGRES_HOST", "custom-postgres")
    monkeypatch.setenv("POSTGRES_USER", "custom-user")
    monkeypatch.setenv("POSTGRES_PASSWORD", "custom-password")
    monkeypatch.setenv("POSTGRES_DB", "custom-db")
    monkeypatch.setenv("REDIS_HOST", "custom-redis")
    monkeypatch.setenv("ELASTIC_URI", "http://custom-es:9200")

    configure_integration_test_env(running_in_docker=True)

    assert os.environ["POSTGRES_HOST"] == "custom-postgres"
    assert os.environ["POSTGRES_USER"] == "custom-user"
    assert os.environ["POSTGRES_PASSWORD"] == "custom-password"
    assert os.environ["POSTGRES_DB"] == "custom-db"
    assert os.environ["REDIS_HOST"] == "custom-redis"
    assert os.environ["ELASTIC_URI"] == "http://custom-es:9200"
