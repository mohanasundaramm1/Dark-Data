The Other 7 Ideas (Good, but maybe secondary)
Hybrid Search (Keywords + Vectors)

Why: Pure vector search creates hallucinations. Adding BM25 (keyword matching) forces the model to respect specific terms.
Reason: High complexity, more "Data Science" than "Data Engineering."
Multimodal Ingestion (OCR/Images) -- You mentioned this!

Why: Companies have scanned PDFs/Imagess. Using a tool like unstructured.io or Tesseract to extract text from images.
Reason: Very valuable, but depends heavily on specific library quality. Can get messy fast.
Streaming Ingestion (Kafka/Redpanda)

Why: Instead of a folder of PDFs, imagine a real-time stream of documents arriving.
Reason: Overkill for a "Dark Data" problem which is usually historic/static archives.
Data Quality Testing (Great Expectations)

Why: automatically fail the pipeline if a chunk is empty or vector dimension is wrong.
Reason: Professional, but distinct from "building features." Good for a "hardening" phase.
Metadata Enrichment (LLM Summarization)

Why: Before embedding, ask a small LLM to "Summarize this chunk". Embed the summary + chunk.
Reason: Increases cost/time significantly. Moves into "AI Engineering" territory.
Deployment (AWS Lambda / ECS)

Why: Get it off your laptop.
Reason: Incurs cloud costs. Harder for recruiters to "run" themselves.
Frontend UI (Streamlit)

Why: Gives you a pretty chat interface to "Talk" to your PDFs.
Reason: It's cool, but it's "Full Stack," not strictly "Data Engineering."