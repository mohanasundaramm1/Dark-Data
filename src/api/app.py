from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import logging
from src.embedding.embedder import get_embedder
from src.config.settings import settings

logger = logging.getLogger(__name__)

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up FastAPI application...")
    # Initialize connection to Qdrant
    if settings.STORAGE_TYPE == "qdrant":
        from qdrant_client import QdrantClient
        app.state.qdrant_client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT, timeout=10)
    else:
        logger.warning("STORAGE_TYPE is not qdrant. Live search is disabled.")
    
    # Initialize embedder once
    logger.info(f"Initializing Embedder: {settings.EMBEDDING_TYPE}")
    app.state.embedder = get_embedder(settings.EMBEDDING_TYPE)
    yield
    # We could close clients here if needed

app = FastAPI(
    title="Dark Data Retrieval API",
    description="API for semantic search over ingested unstructured data.",
    version="1.0.0",
    lifespan=lifespan
)

from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=100)

class SearchResult(BaseModel):
    chunk_id: str
    text: str
    score: float
    metadata: dict



@app.post("/search", response_model=List[SearchResult])
def search(request: QueryRequest):
    if settings.STORAGE_TYPE != "qdrant":
        raise HTTPException(status_code=500, detail="Live search requires Qdrant storage backend.")
        
    try:
        # 1. Embed the query
        embedder = app.state.embedder
        # Assuming embed_documents returns a list of vectors, we grab the first one
        query_vector = embedder.embed_documents([request.query])[0]
        
        # 2. Search Qdrant
        client = app.state.qdrant_client
        if isinstance(query_vector, dict) and "sparse_indices" in query_vector:
            from qdrant_client import models
            prefetch_limit = max(20, request.top_k * 5)
            search_result = client.query_points(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                prefetch=[
                    models.Prefetch(
                        query=query_vector["dense"],
                        using="",
                        limit=prefetch_limit
                    ),
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=query_vector["sparse_indices"],
                            values=query_vector["sparse_values"]
                        ),
                        using="text-sparse",
                        limit=prefetch_limit
                    )
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=request.top_k
            )
        else:
            search_result = client.query_points(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                query=query_vector,
                limit=request.top_k
            )
        
        # 3. Format results
        results = []
        for scored_point in search_result.points:
            results.append(SearchResult(
                chunk_id=str(scored_point.id),
                text=scored_point.payload.get("text", ""),
                score=scored_point.score,
                metadata={k: v for k, v in scored_point.payload.items() if k != "text"}
            ))
            
        return results
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
