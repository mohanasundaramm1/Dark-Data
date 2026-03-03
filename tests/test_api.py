"""
Tests for API authentication and rate-limiting behaviour.
All tests run against the FastAPI TestClient — no live server required.
"""

import os
from fastapi.testclient import TestClient


# Set API_KEY env var BEFORE importing the app so the guard picks it up
os.environ["API_KEY"] = "test-secret-key"
os.environ["EMBEDDING_TYPE"] = "mock"
os.environ["STORAGE_TYPE"] = "parquet"

from src.api.app import app  # noqa: E402

client = TestClient(app)


# ---------------------------------------------------------------------------
# Input validation (no auth needed for health check)
# ---------------------------------------------------------------------------
def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_search_rejects_empty_query():
    resp = client.post(
        "/search",
        json={"query": "", "top_k": 3},
        headers={"X-API-Key": "test-secret-key"},
    )
    assert resp.status_code == 422


def test_search_rejects_top_k_over_limit():
    resp = client.post(
        "/search",
        json={"query": "valid query", "top_k": 999},
        headers={"X-API-Key": "test-secret-key"},
    )
    assert resp.status_code == 422


def test_search_rejects_top_k_zero():
    resp = client.post(
        "/search",
        json={"query": "valid query", "top_k": 0},
        headers={"X-API-Key": "test-secret-key"},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Authentication enforcement
# ---------------------------------------------------------------------------
def test_search_rejects_missing_api_key():
    """When API_KEY is configured, requests without it must get 403."""
    resp = client.post(
        "/search",
        json={"query": "test", "top_k": 3},
        # No X-API-Key header
    )
    assert resp.status_code == 403


def test_search_rejects_wrong_api_key():
    resp = client.post(
        "/search",
        json={"query": "test", "top_k": 3},
        headers={"X-API-Key": "wrong-key"},
    )
    assert resp.status_code == 403


def test_search_accepts_correct_api_key():
    """Correct key + parquet backend → expect 503 (no Qdrant), not 403."""
    resp = client.post(
        "/search",
        json={"query": "test", "top_k": 3},
        headers={"X-API-Key": "test-secret-key"},
    )
    # Should pass auth (not 403) but fail because storage is parquet not qdrant
    assert resp.status_code in (200, 503, 500)
    assert resp.status_code != 403
