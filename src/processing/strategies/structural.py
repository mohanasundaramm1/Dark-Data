from typing import List
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.ingestion.models import IngestedDocument, ProcessedChunk
from src.processing.strategies.base import ChunkingStrategy
from src.config.settings import settings

logger = logging.getLogger(__name__)

class RecursiveStructureStrategy(ChunkingStrategy):
    """
    Intelligently splits text by separators (paragraphs, newlines, sentences) 
    to preserve semantic structure. Best for technical docs or PDFs.
    """
    def __init__(self, chunk_size: int = settings.CHUNK_SIZE, overlap: int = settings.CHUNK_OVERLAP):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ".", " ", ""],
            length_function=len
        )

    def split(self, documents: List[IngestedDocument]) -> List[ProcessedChunk]:
        all_chunks = []
        logger.info(f"Using RecursiveStructureStrategy: Size={self.splitter._chunk_size}, Overlap={self.splitter._chunk_overlap}")

        for doc in documents:
            # LangChain's splitter creates basic strings
            raw_chunks = self.splitter.split_text(doc.content)
            
            for i, chunk_text in enumerate(raw_chunks):
                processed_chunk = ProcessedChunk(
                    parent_doc_id=doc.id,
                    content=chunk_text,
                    chunk_index=i,
                    metadata={**doc.metadata, "strategy": "recursive_structure"}
                )
                all_chunks.append(processed_chunk)
            logger.info(f"Document {doc.filename} split into {len(raw_chunks)} structural chunks.")
        
        return all_chunks
