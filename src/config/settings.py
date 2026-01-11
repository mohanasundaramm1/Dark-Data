import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    DATA_DIR = os.getenv("DATA_DIR", "Data")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", 768))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # Path where processed parquet files (silver layer) will be saved
    OUTPUT_PATH = os.path.join("output", "embeddings.parquet")

settings = Settings()
