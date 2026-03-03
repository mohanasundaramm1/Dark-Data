from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import sys
import os

# Ensure the src directory is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import pipeline functions
from src.config.settings import settings
from src.ingestion.multimodal_loader import MultimodalLoader
from src.processing.factory import StrategyFactory
from src.embedding.embedder import get_embedder
from src.storage.manager import get_storage_manager

# DAG configuration
default_args = {
    'owner': 'data_engineer',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'enterprise_rag_ingestion_pipeline',
    default_args=default_args,
    description='A resilient DAG for unstructured data ingestion, chunking, embedding, and storage.',
    schedule_interval='@daily',
    start_date=days_ago(1),
    catchup=False,
    tags=['ingestion', 'rag', 'dark_data'],
) as dag:

    def load_documents(**context):
        print(f"Loading documents from: {settings.DATA_DIR}")
        loader = MultimodalLoader(settings.DATA_DIR)
        documents = loader.load_documents()
        if not documents:
            raise ValueError("No documents found. Failing pipeline.")
        # In a real Airflow setup, we'd pass data via S3/XCom or save state.
        # For this local demo, we'll return the object references via XCom
        # (Warning: Airflow 2.x supports passing Pydantic objects if pickled, but saving to local intermediate is better)
        # We will do a monolith execution here for simplicity
        return "Documents loaded."

    def process_and_store(**context):
        # 1. Ingestion
        loader = MultimodalLoader(settings.DATA_DIR)
        documents = loader.load_documents()

        # 2. Chunking
        chunker = StrategyFactory.get_strategy(settings.CHUNKING_STRATEGY)
        chunks = chunker.split(documents)

        # 3. Embedding
        embedder = get_embedder(settings.EMBEDDING_TYPE)
        texts = [c.content for c in chunks]
        embeddings = embedder.embed_documents(texts)

        # 4. Storage
        storage = get_storage_manager(settings.STORAGE_TYPE)
        storage.save_embeddings(chunks, embeddings)
        print("Pipeline execution complete.")

    run_etl_task = PythonOperator(
        task_id='run_full_etl',
        python_callable=process_and_store,
    )

    run_etl_task
