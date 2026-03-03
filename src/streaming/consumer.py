import os
import json
import logging
import argparse
from kafka import KafkaConsumer

from src.config.settings import settings
from src.ingestion.multimodal_loader import MultimodalLoader
from src.processing.factory import StrategyFactory
from src.embedding.embedder import get_embedder
from src.storage.manager import get_storage_manager
from src.processing.enricher import SummaryEnricher

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RedpandaConsumer")

KAFKA_BROKER = "localhost:39092"
TOPIC_NAME = "raw_documents"
GROUP_ID = "rag_pipeline_group"

def process_single_file(file_path: str, use_hyde: bool = False):
    logger.info(f"Starting pipeline for single file: {file_path}")
    
    # 1. Ingestion
    loader = MultimodalLoader(os.path.dirname(file_path))
    doc = loader.load_single_document(file_path)
    if not doc:
        logger.error(f"Failed to ingest {file_path}")
        return

    # 2. Chunking
    chunker = StrategyFactory.get_strategy(settings.CHUNKING_STRATEGY)
    chunks = chunker.split([doc])
    
    # 2b. Enrichment (HyDE)
    enricher = SummaryEnricher(use_enrichment=use_hyde)
    chunks = enricher.enrich(chunks)

    # 3. Embedding
    embedder = get_embedder(settings.EMBEDDING_TYPE)
    # Extract text content for embedding (Use Summary if available for HyDE)
    texts_to_embed = [c.summary if c.summary and use_hyde else c.content for c in chunks]
    embeddings = embedder.embed_documents(texts_to_embed)

    # 4. Storage
    storage = get_storage_manager(settings.STORAGE_TYPE)
    storage.save_embeddings(chunks, embeddings)
    
    logger.info(f"Successfully processed and stored {len(chunks)} chunks for {file_path}.")

def consume_events(use_hyde: bool = False):
    logger.info(f"Connecting to Redpanda at {KAFKA_BROKER} to consume from {TOPIC_NAME}")
    
    consumer = KafkaConsumer(
        TOPIC_NAME,
        bootstrap_servers=KAFKA_BROKER,
        group_id=GROUP_ID,
        auto_offset_reset='earliest',
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )

    logger.info("Listening for new files...")
    try:
        for message in consumer:
            data = message.value
            if data.get("event_type") == "PDF_CREATED":
                file_path = data.get("file_path")
                logger.info(f"Received event: {data}")
                process_single_file(file_path, use_hyde=use_hyde)
    except KeyboardInterrupt:
        logger.info("Shutting down consumer...")
    finally:
        consumer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyde", action="store_true", help="Enable LLM Summary Enrichment (HyDE)")
    args = parser.parse_args()
    consume_events(use_hyde=args.hyde)
