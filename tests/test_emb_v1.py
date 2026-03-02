import numpy as np
import pytest
from fastapi.testclient import TestClient

from prokaryotes.emb_v1 import EmbeddingV1

def calculate_cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

@pytest.mark.parametrize("query, related_doc, unrelated_doc", [
    (
        "How do I reset my password?",
        "To change your login credentials, visit the account settings page.",
        "The weather in Seattle is quite rainy today.",
    ),
])
def test_semantic_similarity(query, related_doc, unrelated_doc):
    v1 = EmbeddingV1("Snowflake/snowflake-arctic-embed-l-v2.0")
    with TestClient(v1.app) as client:
        # Get embeddings for query
        payload = {"prompt": "query", "texts": [query]}
        resp = client.post("/emb", json=payload)
        assert resp.status_code == 200, resp
        resp_data = resp.json()
        assert "embeddings" in resp_data, resp_data
        qry_embs = resp_data["embeddings"]

        # Get embeddings for documents
        payload = {"prompt": "document", "texts": [related_doc, unrelated_doc]}
        resp = client.post("/emb", json=payload)
        assert resp.status_code == 200, resp
        resp_data = resp.json()
        assert "embeddings" in resp_data, resp_data
        doc_embs = resp_data["embeddings"]

        score_unrelated = calculate_cosine_similarity(qry_embs[0], doc_embs[1])
        assert score_unrelated < 0.05, "Unrelated documents should have low similarity"

        score_related = calculate_cosine_similarity(qry_embs[0], doc_embs[0])
        assert score_related > 0.35, "Related documents should have higher similarity"
