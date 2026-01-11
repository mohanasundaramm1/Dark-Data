from abc import ABC, abstractmethod
from typing import List
import numpy as np
import logging
from src.config.settings import settings

logger = logging.getLogger(__name__)

class BaseEmbedder(ABC):
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        pass

class MockEmbedder(BaseEmbedder):
    """
    A mock embedder for development and testing pipelines without GPU/API dependency.
    Generates deterministic random vectors based on text length to simulate consistency.
    """
    def __init__(self, dimension: int = settings.EMBEDDING_DIMENSION):
        self.dimension = dimension
        logger.warning(f"Initialized MockEmbedder. This will generate RANDOM vectors of dimension {dimension}.")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        logger.info(f"Generating vectors for {len(texts)} chunks using MockEmbedder.")
        embeddings = []
        for text in texts:
            # We seed random with the length of text to get deterministic "fake" embeddings 
            # (useful for testing pipeline consistency)
            np.random.seed(len(text))
            vector = np.random.rand(self.dimension).tolist()
            embeddings.append(vector)
        return embeddings

def get_embedder(type: str = "mock") -> BaseEmbedder:
    if type == "mock":
        return MockEmbedder()
    raise ValueError(f"Unknown embedder type: {type}")
