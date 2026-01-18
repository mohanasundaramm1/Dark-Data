from abc import ABC, abstractmethod
from typing import List
from src.ingestion.models import IngestedDocument, ProcessedChunk

class ChunkingStrategy(ABC):
    @abstractmethod
    def split(self, documents: List[IngestedDocument]) -> List[ProcessedChunk]:
        """Splits a list of documents into processed chunks."""
        pass
