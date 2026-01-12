import argparse
import logging
import sys
import time

from src.config.settings import settings
from src.ingestion.loader import PDFLoader
from src.processing.text_splitter import TextChunker
from src.embedding.embedder import get_embedder
from src.storage.manager import VectorStorageManager

# Configure Logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def run_pipeline():
    start_time = time.time()
    logger.info("Starting Unstructured Data Ingestion Pipeline...")
    
    # 1. Ingestion
    logger.info("Phase 1: Ingestion")
    loader = PDFLoader(settings.DATA_DIR)
    documents = loader.load_documents()
    if not documents:
        logger.error("No documents found. Exiting.")
        return

    # 2. Processing (Chunking)
    logger.info("Phase 2: Processing (Chunking)")
    chunker = TextChunker()
    chunks = chunker.split(documents)
    logger.info(f"Generated {len(chunks)} total chunks from {len(documents)} documents.")

    # 3. Embedding
    logger.info("Phase 3: Vector Embedding")
    embedder = get_embedder(settings.EMBEDDING_TYPE)
    # Extract text content for embedding
    texts = [c.content for c in chunks]
    embeddings = embedder.embed_documents(texts)
    
    # 4. Storage
    logger.info("Phase 4: Storage")
    storage = VectorStorageManager()
    storage.save_embeddings(chunks, embeddings)
    
    duration = time.time() - start_time
    logger.info(f"Pipeline finished successfully in {duration:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the RAG Data Ingestion Pipeline")
    parser.add_argument("--run", action="store_true", help="Execute the full pipeline")
    
    args = parser.parse_args()
    
    if args.run:
        run_pipeline()
    else:
        parser.print_help()
