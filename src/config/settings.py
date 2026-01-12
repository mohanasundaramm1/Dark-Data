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
    
    # Path where processed parquet files (silver layer) will be saved
    OUTPUT_PATH = os.path.join("output", "embeddings.parquet")

settings = Settings()
