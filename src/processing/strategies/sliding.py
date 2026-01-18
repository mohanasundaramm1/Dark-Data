from typing import List
import logging
from src.ingestion.models import IngestedDocument, ProcessedChunk
from src.processing.strategies.base import ChunkingStrategy

logger = logging.getLogger(__name__)

class SlidingWindowStrategy(ChunkingStrategy):
    """
    Splits text based on word count with a sliding window overlap.
    Ideal for transcripts or continuous speech where context flow is critical.
    """
    def __init__(self, window_size: int = 100, step_size: int = 50):
        # window_size: Number of words per chunk
        # step_size: How many words to move forward (overlap = window_size - step_size)
        self.window_size = window_size
        self.step_size = step_size

    def split(self, documents: List[IngestedDocument]) -> List[ProcessedChunk]:
        all_chunks = []
        logger.info(f"Using SlidingWindowStrategy: Window={self.window_size} words, Step={self.step_size} words")

        for doc in documents:
            words = doc.content.split()
            num_words = len(words)
            chunks = []
            
            if num_words <= self.window_size:
                 chunks.append(doc.content)
            else:
                for i in range(0, num_words, self.step_size):
                    end = i + self.window_size
                    if i + self.window_size > num_words:
                         # Last chunk might be smaller, or we can look back to fill window
                         chunk_words = words[i:]
                    else:
                        chunk_words = words[i:end]
                    
                    if not chunk_words: 
                        break
                        
                    chunks.append(" ".join(chunk_words))
                    
                    if end >= num_words:
                        break

            for i, chunk_text in enumerate(chunks):
                processed_chunk = ProcessedChunk(
                    parent_doc_id=doc.id,
                    content=chunk_text,
                    chunk_index=i,
                    metadata={**doc.metadata, "strategy": "sliding_window"}
                )
                all_chunks.append(processed_chunk)
            logger.info(f"Document {doc.filename} split into {len(chunks)} sliding window chunks.")
        
        return all_chunks
