import logging
from typing import List
from src.ingestion.models import ProcessedChunk

logger = logging.getLogger(__name__)

class SummaryEnricher:
    """
    Implements Hypothetical Document Embeddings (HyDE) by enriching each chunk 
    with an LLM-generated summary. This summary is embedded instead of the raw text
    to improve retrieval accuracy.
    """
    def __init__(self, use_enrichment: bool = False):
        self.use_enrichment = use_enrichment
        self.summarizer = None
        if self.use_enrichment:
            try:
                from transformers import pipeline
                logger.info("Loading lightweight HuggingFace summarizer (distilbart-cnn)...")
                self.summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
                logger.info("Summarizer loaded successfully.")
            except ImportError:
                logger.error("transformers not installed. pip install transformers torch")
                self.use_enrichment = False
            except Exception as e:
                logger.error(f"Failed to load summarizer: {e}")
                self.use_enrichment = False

    def enrich(self, chunks: List[ProcessedChunk]) -> List[ProcessedChunk]:
        if not self.use_enrichment or not self.summarizer:
            return chunks
        
        logger.info(f"Enriching {len(chunks)} chunks with HyDE summaries...")
        
        for i, chunk in enumerate(chunks):
            text = chunk.content
            # Truncate text to avoid model max length errors
            truncated_text = text[:1024] 
            
            try:
                # If text is too short, summarizer might fail
                if len(truncated_text) > 100:
                    res = self.summarizer(truncated_text, max_length=60, min_length=10, do_sample=False)
                    summary_text = res[0]['summary_text'].strip()
                else:
                    summary_text = truncated_text
                
                chunk.summary = summary_text
                chunk.metadata["has_summary"] = True
            except Exception as e:
                logger.warning(f"Summarization failed for chunk {chunk.chunk_id}, skipping: {e}")
                chunk.summary = truncated_text
                chunk.metadata["has_summary"] = False
                
        logger.info("HyDE enrichment completed.")
        return chunks
