from typing import List
import logging
from src.ingestion.models import IngestedDocument, ProcessedChunk
from src.processing.strategies.base import ChunkingStrategy
from src.config.settings import settings

logger = logging.getLogger(__name__)

class FixedSizeStrategy(ChunkingStrategy):
    """
    Splits text into chunks of a specified fixed character size with overlap.
    """
    def __init__(self, chunk_size: int = settings.CHUNK_SIZE, overlap: int = settings.CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, documents: List[IngestedDocument]) -> List[ProcessedChunk]:
        all_chunks = []
        logger.info(f"Using FixedSizeStrategy: Size={self.chunk_size}, Overlap={self.overlap}")

        for doc in documents:
            chunks = self._chunk_text(doc.content)
            for i, chunk_text in enumerate(chunks):
                processed_chunk = ProcessedChunk(
                    parent_doc_id=doc.id,
                    content=chunk_text,
                    chunk_index=i,
                    metadata=doc.metadata
                )
                all_chunks.append(processed_chunk)
            logger.info(f"Document {doc.filename} split into {len(chunks)} chunks.")
        
        return all_chunks

    def _chunk_text(self, text: str) -> List[str]:
        if not text:
            return []
        
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            
            if end >= text_len:
                break
                
            start += (self.chunk_size - self.overlap)
        
        return chunks
