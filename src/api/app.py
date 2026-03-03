from fastapi import FastAPI, HTTPException, Security, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from prometheus_fastapi_instrumentator import Instrumentator
from typing import List
import logging
import os
from starlette.responses import Response
from src.embedding.embedder import get_embedder
from src.config.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate Limiter
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])

# ---------------------------------------------------------------------------
# API Key Security (Optional: enable by setting API_KEY in .env)
# ---------------------------------------------------------------------------
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

_API_KEY = os.getenv("API_KEY", "")  # Empty = no auth required (dev mode)


def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    if not _API_KEY:
        return "dev-mode"  # No key configured → open access
    if api_key != _API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API Key. Set X-API-Key header.")
    return api_key


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up FastAPI application...")
    if settings.STORAGE_TYPE == "qdrant":
        from qdrant_client import QdrantClient
        app.state.qdrant_client = QdrantClient(
            host=settings.QDRANT_HOST, port=settings.QDRANT_PORT, timeout=10
        )
    else:
        logger.warning("STORAGE_TYPE is not qdrant. Live search is disabled.")

    logger.info(f"Initializing Embedder: {settings.EMBEDDING_TYPE}")
    app.state.embedder = get_embedder(settings.EMBEDDING_TYPE)
    yield


# ---------------------------------------------------------------------------
# App Construction
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Dark Data Retrieval API",
    description="Semantic RAG search over ingested unstructured enterprise data.",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS Policy
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:8501").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Rate Limiting
app.state.limiter = limiter


def rate_limit_exception_handler(request: Request, exc: Exception) -> Response:
    if isinstance(exc, RateLimitExceeded):
        return _rate_limit_exceeded_handler(request, exc)
    return Response(status_code=500)


app.add_exception_handler(RateLimitExceeded, rate_limit_exception_handler)

# Prometheus Metrics  (auto-exposes /metrics endpoint)
Instrumentator().instrument(app).expose(app)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=100)


class SearchResult(BaseModel):
    chunk_id: str
    text: str
    score: float
    metadata: dict


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health_check():
    return {"status": "ok", "version": "2.0.0"}


@app.post("/search", response_model=List[SearchResult])
@limiter.limit("30/minute")
def search(
    request: Request,
    body: QueryRequest,
    _: str = Security(verify_api_key),
) -> list:
    if settings.STORAGE_TYPE != "qdrant":
        raise HTTPException(status_code=503, detail="Live search requires Qdrant storage backend.")

    try:
        embedder = app.state.embedder
        query_vector = embedder.embed_documents([body.query])[0]

        client = app.state.qdrant_client
        if isinstance(query_vector, dict) and "sparse_indices" in query_vector:
            from qdrant_client import models
            prefetch_limit = max(20, body.top_k * 5)
            search_result = client.query_points(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                prefetch=[
                    models.Prefetch(query=query_vector["dense"], using="", limit=prefetch_limit),
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=query_vector["sparse_indices"],
                            values=query_vector["sparse_values"],
                        ),
                        using="text-sparse",
                        limit=prefetch_limit,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=body.top_k,
            )
        else:
            search_result = client.query_points(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                query=query_vector,
                limit=body.top_k,
            )

        return [
            SearchResult(
                chunk_id=str(p.id),
                text=p.payload.get("text", ""),
                score=p.score,
                metadata={k: v for k, v in p.payload.items() if k != "text"},
            )
            for p in search_result.points
        ]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
