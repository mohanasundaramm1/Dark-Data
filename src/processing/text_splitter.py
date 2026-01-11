from typing import List
from src.ingestion.models import IngestedDocument, ProcessedChunk
from src.config.settings import settings
import logging

logger = logging.getLogger(__name__)

class TextChunker:
    """
    Custom implementation of credential-aware text chunking.
    Splits text into chunks of a specified size with overlap.
    """
    def __init__(self, chunk_size: int = settings.CHUNK_SIZE, overlap: int = settings.CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, documents: List[IngestedDocument]) -> List[ProcessedChunk]:
        all_chunks = []
        logger.info(f"Starting chunking process. Strategy: Fixed Size ({self.chunk_size}) with Overlap ({self.overlap})")

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
        """Simple sliding window chunking."""
        if not text:
            return []
        
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            
            # If we reached the end, break
            if end >= text_len:
                break
                
            # Move start pointer forward, accounting for overlap
            start += (self.chunk_size - self.overlap)
        
        return chunks
