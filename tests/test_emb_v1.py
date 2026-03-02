import numpy as np
import pytest
from fastapi.testclient import TestClient

from prokaryotes.models_v1 import TextEmbeddingResponse
from prokaryotes.emb_v1 import EmbeddingV1

@pytest.fixture(scope="session")
def client():
    v1 = EmbeddingV1("Snowflake/snowflake-arctic-embed-l-v2.0")
    with TestClient(v1.app) as c:
        yield c

@pytest.mark.parametrize(", ".join([
    "query",
    "related_doc",
    "related_doc_threshold",
    "unrelated_doc",
    "unrelated_doc_threshold",
    "truncate_to"
]), [
    (
            "How do I reset my password?",
            "To change your login credentials, visit the account settings page.",
            0.4,
            "The weather in Seattle is quite rainy today.",
            0.005,
            256,
    ),
    (
            "How can I lower my monthly energy bills?",
            "Tips for reducing residential electricity consumption and heating costs.",
            0.45,
            "Monthly energy reports are available in the administrative dashboard.",
            0.35,
            256,
    ),
    (
            "How to handle null values in a pandas dataframe?",
            "Use the dropna method or fillna with a specific value to clean your data.",
            0.45,
            "The panda is a bear native to south central China.",
            0.13,
            256,
    ),
    (
            "Why is my account not activated?",
            "Common reasons for pending activation include missing email verification.",
            0.55,
            "Your account is successfully activated and ready for use.",
            0.53,
            256,
    ),
])
def test_semantic_similarity(
        client,
        query,
        related_doc,
        related_doc_threshold,
        unrelated_doc,
        unrelated_doc_threshold,
        truncate_to,
):
    v1 = EmbeddingV1("Snowflake/snowflake-arctic-embed-l-v2.0")
    with TestClient(v1.app) as client:
        # Get embeddings for query
        payload = {"prompt": "query", "texts": [query], "truncate_to": truncate_to}
        resp = client.post("/emb", json=payload)
        assert resp.status_code == 200, resp
        qry_embs = TextEmbeddingResponse.model_validate(resp.json()).embeddings

        # Get embeddings for documents
        payload = {"prompt": "document", "texts": [related_doc, unrelated_doc], "truncate_to": truncate_to}
        resp = client.post("/emb", json=payload)
        assert resp.status_code == 200, resp
        doc_embs = TextEmbeddingResponse.model_validate(resp.json()).embeddings

        score_unrelated = np.dot(qry_embs[0], doc_embs[1])
        assert score_unrelated < unrelated_doc_threshold, "Unrelated documents should have low similarity"

        score_related = np.dot(qry_embs[0], doc_embs[0])
        assert score_related > related_doc_threshold, "Related documents should have higher similarity"
