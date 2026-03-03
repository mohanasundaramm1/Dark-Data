from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import logging
from src.embedding.embedder import get_embedder
from src.config.settings import settings

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Dark Data Retrieval API",
    description="API for semantic search over ingested unstructured data.",
    version="1.0.0"
)

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

class SearchResult(BaseModel):
    chunk_id: str
    text: str
    score: float
    metadata: dict

@app.on_event("startup")
def startup_event():
    logger.info("Starting up FastAPI application...")
    # Initialize connection to Qdrant
    if settings.STORAGE_TYPE == "qdrant":
        from qdrant_client import QdrantClient
        app.state.qdrant_client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
    else:
        logger.warning("STORAGE_TYPE is not qdrant. Live search is disabled.")

@app.post("/search", response_model=List[SearchResult])
def search(request: QueryRequest):
    if settings.STORAGE_TYPE != "qdrant":
        raise HTTPException(status_code=500, detail="Live search requires Qdrant storage backend.")
        
    try:
        # 1. Embed the query
        embedder = get_embedder(settings.EMBEDDING_TYPE)
        # Assuming embed_documents returns a list of vectors, we grab the first one
        query_vector = embedder.embed_documents([request.query])[0]
        
        # 2. Search Qdrant
        client = app.state.qdrant_client
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
