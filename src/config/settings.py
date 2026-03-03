import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    DATA_DIR = os.getenv("DATA_DIR", "Data")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", 768))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # Embedding Configuration
    # Options: "mock", "huggingface"
    EMBEDDING_TYPE = os.getenv("EMBEDDING_TYPE", "mock")
    # Model name for HuggingFace (e.g., "all-mpnet-base-v2")
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-mpnet-base-v2")
    
    # Chunking Configuration
    # Options: "fixed", "sliding", "structural"
    CHUNKING_STRATEGY = os.getenv("CHUNKING_STRATEGY", "structural")

    # Storage Configuration
    # Options: "parquet", "qdrant"
    STORAGE_TYPE = os.getenv("STORAGE_TYPE", "qdrant")
    
    # Path where processed parquet files (silver layer) will be saved (used if STORAGE_TYPE='parquet')
    OUTPUT_PATH = os.path.join("output", "embeddings.parquet")

    # Qdrant Config
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
    QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "dark_data")

settings = Settings()
