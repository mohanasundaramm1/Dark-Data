from abc import ABC, abstractmethod
from typing import List
import pandas as pd
import os
import logging
from src.ingestion.models import ProcessedChunk
from src.config.settings import settings

logger = logging.getLogger(__name__)

class BaseStorageManager(ABC):
    @abstractmethod
    def save_embeddings(self, chunks: List[ProcessedChunk], embeddings: List[List[float]]):
        """Saves chunks and embeddings to the underlying storage system."""
        pass

class ParquetStorageManager(BaseStorageManager):
    """Saves vectorized chunks to a Parquet file (Data Lake / Silver Layer)."""
    def save_embeddings(self, chunks: List[ProcessedChunk], embeddings: List[List[float]]):
        if not chunks:
            logger.warning("No chunks to save.")
            return

        logger.info(f"Preparing to save {len(chunks)} vectors to Parquet Storage.")
        
        data = []
        for chunk, vector in zip(chunks, embeddings):
            row = {
                "chunk_id": chunk.chunk_id,
                "parent_id": chunk.parent_doc_id,
                "text": chunk.content,
                "vector": vector,
                "metadata": chunk.metadata
            }
            data.append(row)
            
        df = pd.DataFrame(data)
        
        output_dir = os.path.dirname(settings.OUTPUT_PATH)
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Writing parquet file to {settings.OUTPUT_PATH}...")
        try:
            df.to_parquet(settings.OUTPUT_PATH, index=False)
            logger.info("Successfully persisted vector data to Parquet.")
        except Exception as e:
            logger.error(f"Failed to save parquet file: {e}")
            raise

class QdrantStorageManager(BaseStorageManager):
    """Saves vectorized chunks to a live Qdrant Vector Database (Serving Layer)."""
    def __init__(self):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import VectorParams, Distance
            
            logger.info(f"Connecting to Qdrant at {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
            self.client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
            
            # Ensure the collection exists
            collections_response = self.client.get_collections()
            collection_names = [c.name for c in collections_response.collections]
            
            if settings.QDRANT_COLLECTION_NAME not in collection_names:
                logger.info(f"Creating new Qdrant collection: '{settings.QDRANT_COLLECTION_NAME}'")
                self.client.create_collection(
                    collection_name=settings.QDRANT_COLLECTION_NAME,
                    vectors_config=VectorParams(size=settings.EMBEDDING_DIMENSION, distance=Distance.COSINE),
                )
            else:
                logger.info(f"Using existing Qdrant collection: '{settings.QDRANT_COLLECTION_NAME}'")
                
        except ImportError:
            logger.error("qdrant-client not installed. Please run: pip install qdrant-client")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise

    def save_embeddings(self, chunks: List[ProcessedChunk], embeddings: List[List[float]]):
        if not chunks:
            logger.warning("No chunks to save.")
            return

        from qdrant_client.models import PointStruct

        logger.info(f"Preparing to save {len(chunks)} vectors to Qdrant Vector DB.")
        
        points = []
        for chunk, vector in zip(chunks, embeddings):
            point = PointStruct(
                id=chunk.chunk_id,
                vector=vector,
                payload={
                    "parent_id": chunk.parent_doc_id,
                    "text": chunk.content,
                    **chunk.metadata
                }
            )
            points.append(point)
            
        try:
            self.client.upsert(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                points=points
            )
            logger.info(f"Successfully upserted {len(points)} vectors to Qdrant collection '{settings.QDRANT_COLLECTION_NAME}'.")
        except Exception as e:
            logger.error(f"Failed to upsert vectors to Qdrant: {e}")
            raise


def get_storage_manager(storage_type: str = settings.STORAGE_TYPE) -> BaseStorageManager:
    """Factory to return the appropriate Storage Manager."""
    if storage_type == "parquet":
        return ParquetStorageManager()
    elif storage_type == "qdrant":
        return QdrantStorageManager()
    else:
        raise ValueError(f"Unknown storage type: {storage_type}")
