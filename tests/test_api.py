from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)

# We might not have a running Qdrant instance for the test. We can mock it or just test input validation.

def test_api_input_validation():
    # Attempt to search with empty query
    response = client.post("/search", json={"query": "", "top_k": 3})
    assert response.status_code == 422 # Validation Error

    # Attempt to search with negative top_k
    response = client.post("/search", json={"query": "test query", "top_k": -1})
    assert response.status_code == 422

    # Attempt to search with too large top_k
    response = client.post("/search", json={"query": "test query", "top_k": 200})
    assert response.status_code == 422

# Testing actual retrieval requires mock of qdrant or a running service.
