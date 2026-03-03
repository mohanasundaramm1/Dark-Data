from abc import ABC, abstractmethod
from typing import List, Any
import pandas as pd
import os
import logging
from src.ingestion.models import ProcessedChunk
from src.config.settings import settings

logger = logging.getLogger(__name__)

class BaseStorageManager(ABC):
    @abstractmethod
    def save_embeddings(self, chunks: List[ProcessedChunk], embeddings: List[Any]):
        """Saves chunks and embeddings to the underlying storage system."""
        pass

    def validate_data_quality(self, chunks: List[ProcessedChunk], embeddings: List[Any]):
        import great_expectations as gx
        
        logger.info("Running Great Expectations Data Quality Checks...")
        
        data = []
        for chunk, vector_data in zip(chunks, embeddings):
            row = {
                "chunk_id": chunk.chunk_id,
                "parent_id": chunk.parent_doc_id,
                "text": chunk.content,
            }
            if isinstance(vector_data, dict) and "dense" in vector_data:
                row["vector_len"] = len(vector_data["dense"])
            elif isinstance(vector_data, list):
                row["vector_len"] = len(vector_data)
            else:
                row["vector_len"] = 0
            data.append(row)
            
        df = pd.DataFrame(data)
        context = gx.get_context(mode="ephemeral")
        
        # Connect to data
        data_source = context.data_sources.add_pandas("memory_data")
        data_asset = data_source.add_dataframe_asset("chunks_asset")
        batch_def = data_asset.add_batch_definition_whole_dataframe("my_batch_def")
        
        # Create an Expectation Suite
        suite = context.suites.add(gx.ExpectationSuite(name="pipeline_checks"))
        
        # 1. Content must not be null
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(column="text")
        )
        # 2. Vector must be length 768
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column="vector_len", min_value=768, max_value=768
            )
        )
        # 3. ID format
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(column="chunk_id")
        )

        validation_result = context.validation_definitions.add(
            gx.ValidationDefinition(
                name="pipeline_validation",
                data=batch_def,
                suite=suite
            )
        ).run(batch_parameters={"dataframe": df})
        
        if not validation_result.success:
            logger.error("Data Quality Data Contract FAILED!")
            raise Exception(f"Data Quality Assertions Failed: {validation_result}")
        logger.info("Data Quality Rules Passed Successfully!")

class ParquetStorageManager(BaseStorageManager):
    """Saves vectorized chunks to a Parquet file (Data Lake / Silver Layer)."""
    def save_embeddings(self, chunks: List[ProcessedChunk], embeddings: List[Any]):
        if not chunks:
            logger.warning("No chunks to save.")
            return

        self.validate_data_quality(chunks, embeddings)
        
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
            from qdrant_client.models import VectorParams, Distance, SparseVectorParams
            
            logger.info(f"Connecting to Qdrant at {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
            self.client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
            
            # Ensure the collection exists
            collections_response = self.client.get_collections()
            collection_names = [c.name for c in collections_response.collections]
            
            if settings.QDRANT_COLLECTION_NAME not in collection_names:
                logger.info(f"Creating new Qdrant collection: '{settings.QDRANT_COLLECTION_NAME}'")
                kwargs = {
                    "collection_name": settings.QDRANT_COLLECTION_NAME,
                    "vectors_config": VectorParams(size=settings.EMBEDDING_DIMENSION, distance=Distance.COSINE),
                }
                if settings.EMBEDDING_TYPE == "hybrid":
                    kwargs["sparse_vectors_config"] = {"text-sparse": SparseVectorParams()}
                self.client.create_collection(**kwargs)
            else:
                logger.info(f"Using existing Qdrant collection: '{settings.QDRANT_COLLECTION_NAME}'")
                
        except ImportError:
            logger.error("qdrant-client not installed. Please run: pip install qdrant-client")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise

    def save_embeddings(self, chunks: List[ProcessedChunk], embeddings: List[Any]):
        if not chunks:
            logger.warning("No chunks to save.")
            return

        self.validate_data_quality(chunks, embeddings)

        from qdrant_client.models import PointStruct, SparseVector

        logger.info(f"Preparing to save {len(chunks)} vectors to Qdrant Vector DB.")
        
        points = []
        for chunk, vector_data in zip(chunks, embeddings):
            if isinstance(vector_data, dict) and "sparse_indices" in vector_data:
                # Hybrid format
                vector = {
                    "": vector_data["dense"],
                    "text-sparse": SparseVector(
                        indices=vector_data["sparse_indices"],
                        values=vector_data["sparse_values"]
                    )
                }
            else:
                vector = vector_data # List of floats
                
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
