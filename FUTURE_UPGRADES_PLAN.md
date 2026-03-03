# Detailed Micro-Level Implementation Plan: Phase 2 Pipeline Upgrades

This document outlines the detailed, step-by-step technical blueprints required to implement the remaining 7 advanced Data Engineering and AI features into the existing unstructured data pipeline.

---

## 1. Hybrid Search (Keywords + Vectors)
**Objective:** Combine semantic understanding (dense vectors) with exact keyword matching (sparse vectors/BM25) to prevent hallucinations and improve precision.

**Implementation Steps:**
1.  **Dependencies:** `qdrant-client` already supports this, but we need `fastembed` for generating sparse vectors locally.
    *   `pip install fastembed`
2.  **Modify `src/embedding/embedder.py`:**
    *   Initialize a sparse embedding model alongside the dense model (e.g., `SparseTextEmbedding("Qdrant/bm25")`).
    *   Update `embed_documents` to return a `Tuple[List[List[float]], List[Dict]]` representing dense vectors and sparse vectors (indices and values).
3.  **Update Qdrant Schema (`src/storage/manager.py`):**
    *   In the collection creation logic, define sparse vector parameters alongside dense parameters.
    ```python
    self.client.create_collection(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        vectors_config={"text-dense": VectorParams(size=768, distance=Distance.COSINE)},
        sparse_vectors_config={"text-sparse": SparseVectorParams()}
    )
    ```
4.  **Update API Layer (`src/api/app.py`):**
    *   Update the `client.query_points` call to accept both dense and sparse representations of the query.
    *   Use Qdrant's Reciprocal Rank Fusion (RRF) to blend the scores from both vector spaces.

---

## 2. Multimodal Ingestion (OCR/Images)
**Objective:** Extract text and tables from non-text PDFs (scans) and raw images.

**Implementation Steps:**
1.  **Dependencies:** Introduce `unstructured` and `pytesseract`.
    *   `pip install "unstructured[all-docs]" pytesseract`
    *   Requires system-level dependencies: `brew install tesseract poppler`
2.  **Create `src/ingestion/multimodal_loader.py`:**
    *   Implement an `UnstructuredLoader` class that replaces `PyPDFLoader`.
    *   Configure it to route files: unstructured partitions text by element type (Title, NarrativeText, Title, Image).
3.  **Route to Strategy:**
    *   Images/Scanned PDFs detected by the loader will be run through Tesseract OCR.
    *   The resulting text is packaged into the existing `IngestedDocument` Pydantic model.
4.  **Metadata Tracking:** 
    *   Add `{"source_type": "scanned_pdf" | "image"}` to the metadata dictionary to track data lineage.

---

## 3. Streaming Ingestion (Kafka/Redpanda)
**Objective:** Decouple ingestion from processing by setting up a pub/sub event stream.

**Implementation Steps:**
1.  **Infrastructure:** Add Kafka or Redpanda (lighter) to `docker-compose.yml`.
    ```yaml
    redpanda:
      image: docker.redpanda.com/redpandadata/redpanda:latest
      ports: ["9092:9092"]
    ```
2.  **Create Producer (`src/ingestion/producer.py`):**
    *   Write a Python script using `confluent-kafka` that monitors `Data/` (using Python `watchdog`).
    *   When a new PDF is dropped in the folder, the producer reads the file bytes and pushes a message to a `raw_documents` Kafka topic.
3.  **Create Consumer (`src/processing/consumer.py`):**
    *   A constantly running worker process that listens to `raw_documents`.
    *   Upon receiving an event, it deserializes the byte stream, triggers the Chunking Strategy, generates embeddings, and pushes to Qdrant.
4.  **Airflow Refactor:** Airflow transitions from "batch processing everything at once" to scheduling the health checks of the streaming containers.

---

## 4. Data Quality Testing (Great Expectations)
**Objective:** Ensure pipeline reliability by adding programmatic data contracts (assertions).

**Implementation Steps:**
1.  **Dependencies:** `pip install great_expectations`
2.  **Initialize Context:** Run `great_expectations init` in the root directory.
3.  **Define Expectations (The Data Contract):**
    *   *Expectation 1:* The `vector` column must be exactly length 768.
    *   *Expectation 2:* The `content` column must not be null or empty.
    *   *Expectation 3:* `parent_doc_id` must be a valid UUID.
4.  **Integrate with Pipeline (`src/storage/manager.py`):**
    *   Before uploading points to Qdrant (or writing to Parquet), run the DataFrame through the G.E. Checkpoint.
    *   If the checkpoint fails, intentionally crash the Airflow task (`raise DataQualityException`) so bad data never leaks into the Serving Layer.

---

## 5. Metadata Enrichment (LLM Summarization)
**Objective:** Generate a "summary" for every chunk to provide better context during vector retrieval.

**Implementation Steps:**
1.  **Dependencies:** We already have `langchain`. We need to integrate `ollama` for local inference.
    *   `pip install langchain-community`
    *   Ensure Ollama app is running locally with a model (e.g., `ollama run llama3`).
2.  **Create `src/processing/enricher.py`:**
    *   After chunks are created by the `ChunkingStrategy`, pass them to the enricher.
    *   For each chunk, prompt the LLM: *"Summarize the following text in 1 sentence: {chunk.content}"*
3.  **Modify Schema:**
    *   Update `ProcessedChunk` Pydantic model to include `summary: Optional[str]`.
4.  **Embed and Store:**
    *   *Crucial Change:* Generate the embedding vector based on the **LLM Summary**, not the raw text.
    *   Store both the raw text and summary in Qdrant's payload. This is called the "Hypothetical Document Embedding" (HyDE) pattern.

---

## 6. Deployment (AWS Lambda / ECS)
**Objective:** Move the architecture to the cloud using Infrastructure as Code (Terraform).

**Implementation Steps:**
1.  **Dependencies:** `brew install terraform awscli`
2.  **Containerize Pipeline:** 
    *   Create a production `Dockerfile` that packages the Airflow DAGs, FastAPI app, and our `src/` code.
3.  **Terraform Configuration (`infra/main.tf`):**
    *   *Compute:* Define an AWS ECS Fargate cluster to run the FastAPI app and Airflow workers statelessly.
    *   *Storage:* Define an AWS S3 bucket for the `Data/` lake.
    *   *Database:* Define an AWS RDS instance for Airflow metadata. (Note: Qdrant would switch to Qdrant Cloud or be deployed separately on EC2).
4.  **CI/CD Pipeline (`.github/workflows/deploy.yml`):**
    *   On push to `main` branch: GitHub Actions builds the Docker image, pushes it to AWS ECR, and forces ECS to deploy the new container.

---

## 7. Frontend UI (Streamlit)
**Objective:** Provide a graphical chat interface for stakeholders to query the Dark Data.

**Implementation Steps:**
1.  **Dependencies:** `pip install streamlit`
2.  **Create `src/ui/app.py`:**
    *   Build a Chatbot UI using `st.chat_message` and `st.chat_input`.
3.  **Connect to API:**
    *   When the user types a message, `app.py` makes an HTTP POST request to our FastAPI backend (`http://localhost:8888/search`).
    *   Extract the returned chunks from the API response payload.
4.  **Synthesize Answer:**
    *   Pass the retrieved chunks + the user's question to a local LLM (Ollama) to generate a conversational answer ("Based on the provided documents...").
5.  **Run:** `streamlit run src/ui/app.py`
