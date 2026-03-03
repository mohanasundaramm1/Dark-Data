import os
import json
import logging
import argparse
import traceback
from kafka import KafkaConsumer, KafkaProducer

from src.config.settings import settings
from src.ingestion.multimodal_loader import MultimodalLoader
from src.processing.factory import StrategyFactory
from src.embedding.embedder import get_embedder, BaseEmbedder
from src.storage.manager import get_storage_manager, BaseStorageManager
from src.processing.enricher import SummaryEnricher

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RedpandaConsumer")

KAFKA_BROKER = settings.KAFKA_BROKER
TOPIC_NAME = settings.KAFKA_TOPIC_NAME
GROUP_ID = settings.KAFKA_GROUP_ID


# =============================================================================
# ProcessContext: expensive singletons initialised ONCE per process
# =============================================================================
class ProcessContext:
    """
    Holds all expensive, re-usable singletons for the consumer process.
    Initialised once at startup; each message handler receives it by reference.
    """
    def __init__(self, use_hyde: bool = False) -> None:
        logger.info("Initialising ProcessContext (one-time startup)...")

        self.use_hyde = use_hyde
        self.chunker = StrategyFactory.get_strategy(settings.CHUNKING_STRATEGY)
        self.embedder: BaseEmbedder = get_embedder(settings.EMBEDDING_TYPE)
        self.storage: BaseStorageManager = get_storage_manager(settings.STORAGE_TYPE)
        self.enricher = SummaryEnricher(use_enrichment=use_hyde)

        logger.info("ProcessContext ready.")


def process_single_file(file_path: str, ctx: ProcessContext) -> None:
    """Process one PDF through the full pipeline using pre-warmed singletons."""
    logger.info(f"Processing: {file_path}")

    # 1. Ingestion
    loader = MultimodalLoader(os.path.dirname(file_path))
    doc = loader.load_single_document(file_path)
    if not doc:
        raise ValueError(f"Failed to ingest {file_path}")

    # 2. Chunking
    chunks = ctx.chunker.split([doc])

    # 3. Optional HyDE enrichment
    if ctx.use_hyde:
        chunks = ctx.enricher.enrich(chunks)

    # 4. Embedding  (use summary text if HyDE is on)
    texts_to_embed = [
        c.summary if c.summary and ctx.use_hyde else c.content for c in chunks
    ]
    embeddings = ctx.embedder.embed_documents(texts_to_embed)

    # 5. Storage
    ctx.storage.save_embeddings(chunks, embeddings)

    logger.info(f"Stored {len(chunks)} chunks for {os.path.basename(file_path)}.")


def consume_events(use_hyde: bool = False) -> None:
    logger.info(f"Connecting to Redpanda at {KAFKA_BROKER} → topic '{TOPIC_NAME}'")

    # ------------------------------------------------------------------
    # One-time initialisation of all expensive pipeline components
    # ------------------------------------------------------------------
    ctx = ProcessContext(use_hyde=use_hyde)

    # Reliability: Manual Commits, no auto-commit
    consumer = KafkaConsumer(
        TOPIC_NAME,
        bootstrap_servers=KAFKA_BROKER,
        group_id=GROUP_ID,
        auto_offset_reset='earliest',
        enable_auto_commit=False,
        value_deserializer=lambda x: json.loads(x.decode('utf-8')),
    )

    # Reliability: Dead-Letter Queue producer
    dlq_producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    )
    dlq_topic = f"{TOPIC_NAME}_dlq"

    logger.info("Consumer ready — listening with strict reliability mode...")
    try:
        for message in consumer:
            data = message.value
            try:
                if data.get("event_type") == "PDF_CREATED":
                    file_path = data.get("file_path", "")
                    logger.info(f"Event received: {data}")
                    process_single_file(file_path, ctx)

                # Manual commit only after successful handling
                consumer.commit()
                logger.debug(f"Committed offset {message.offset}")

            except Exception as exc:
                logger.error(f"Failed at offset {message.offset}: {exc}")

                # Poison-message → DLQ (never block the consumer on bad messages)
                dlq_message = {
                    "original_event": data,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "failed_at_offset": message.offset,
                }
                dlq_producer.send(dlq_topic, value=dlq_message)
                dlq_producer.flush()
                logger.info(f"Poison message forwarded to DLQ topic '{dlq_topic}'")

                # Commit anyway so we don't replay poison messages forever
                consumer.commit()

    except KeyboardInterrupt:
        logger.info("Shutting down consumer gracefully...")
    finally:
        consumer.close()
        dlq_producer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyde", action="store_true", help="Enable HyDE LLM enrichment")
    args = parser.parse_args()
    consume_events(use_hyde=args.hyde)
