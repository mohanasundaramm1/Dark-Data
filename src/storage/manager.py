import pandas as pd
import logging
import os
from typing import List
from src.ingestion.models import ProcessedChunk
from src.config.settings import settings

logger = logging.getLogger(__name__)

class VectorStorageManager:
    """
    Manages the storage of processed chunks and their embeddings.
    Currently supports exporting to Parquet for Bronze/Silver layer storage.
    """
    def save_embeddings(self, chunks: List[ProcessedChunk], embeddings: List[List[float]]):
        if not chunks:
            logger.warning("No chunks to save.")
            return

        logger.info(f"Preparing to save {len(chunks)} vectors to storage.")
        
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
            logger.info("Successfully persisted vector data.")
        except Exception as e:
            logger.error(f"Failed to save parquet file: {e}")
            raise
