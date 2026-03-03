import os
import time
import json
import logging
import argparse
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from kafka import KafkaProducer

from src.config.settings import settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RedpandaProducer")

KAFKA_BROKER = settings.KAFKA_BROKER
TOPIC_NAME = settings.KAFKA_TOPIC_NAME

# Initialize Kafka Producer
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

class PDFHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.pdf'):
            logger.info(f"New PDF detected: {event.src_path}")
            self.publish_to_kafka(event.src_path)

    def publish_to_kafka(self, file_path):
        try:
            filename = os.path.basename(file_path)
            message = {
                "event_type": "PDF_CREATED",
                "file_path": file_path,
                "filename": filename,
                "timestamp": time.time()
            }
            producer.send(TOPIC_NAME, value=message)
            producer.flush()
            logger.info(f"Successfully published event for {filename} to {TOPIC_NAME}")
        except Exception as e:
            logger.error(f"Failed to publish to Kafka: {e}")

def watch_directory(directory: str):
    logger.info(f"Starting directory watcher on: {directory}")
    logger.info(f"Connecting to Redpanda at {KAFKA_BROKER}")
    
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    event_handler = PDFHandler()
    observer = Observer()
    observer.schedule(event_handler, path=directory, recursive=False)
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=settings.DATA_DIR, help="Directory to watch")
    args = parser.parse_args()
    watch_directory(args.dir)
