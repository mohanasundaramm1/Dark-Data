"""
Tests for the streaming consumer's ProcessContext and process_single_file logic.
Uses mocking — no live Kafka/Qdrant required.
"""

import pytest
from unittest.mock import MagicMock, patch
from src.ingestion.models import IngestedDocument, ProcessedChunk


# -------------------------------------------------------------------------
# ProcessContext initialisation
# -------------------------------------------------------------------------
def test_process_context_initialises_all_components():
    """ProcessContext must wire up chunker, embedder, storage, and enricher."""
    with (
        patch("src.streaming.consumer.StrategyFactory") as mock_factory,
        patch("src.streaming.consumer.get_embedder") as mock_embedder_fn,
        patch("src.streaming.consumer.get_storage_manager") as mock_storage_fn,
        patch("src.streaming.consumer.SummaryEnricher") as mock_enricher_class,
    ):
        from src.streaming.consumer import ProcessContext

        _ = ProcessContext(use_hyde=False)

        mock_factory.get_strategy.assert_called_once()
        mock_embedder_fn.assert_called_once()
        mock_storage_fn.assert_called_once()
        mock_enricher_class.assert_called_once_with(use_enrichment=False)


def test_process_context_passes_use_hyde_to_enricher():
    with (
        patch("src.streaming.consumer.StrategyFactory"),
        patch("src.streaming.consumer.get_embedder"),
        patch("src.streaming.consumer.get_storage_manager"),
        patch("src.streaming.consumer.SummaryEnricher") as mock_enricher_class,
    ):
        from src.streaming.consumer import ProcessContext

        ProcessContext(use_hyde=True)
        mock_enricher_class.assert_called_once_with(use_enrichment=True)


# -------------------------------------------------------------------------
# process_single_file — full pipeline wiring via mocked context
# -------------------------------------------------------------------------
def test_process_single_file_raises_on_missing_doc():
    """If the loader fails to produce a document, a ValueError must be raised."""
    fake_ctx = MagicMock()

    with (
        patch("src.streaming.consumer.MultimodalLoader") as mock_loader_class,
    ):
        mock_loader = MagicMock()
        mock_loader.load_single_document.return_value = None
        mock_loader_class.return_value = mock_loader

        from src.streaming.consumer import process_single_file

        with pytest.raises(ValueError, match="Failed to ingest"):
            process_single_file("/nonexistent/file.pdf", fake_ctx)


def test_process_single_file_calls_storage_save():
    """Happy path: storage.save_embeddings must be called once."""
    fake_doc = IngestedDocument(filename="t.pdf", content="hello world")
    fake_chunk = ProcessedChunk(parent_doc_id=fake_doc.id, content="hello world", chunk_index=0)

    fake_ctx = MagicMock()
    fake_ctx.use_hyde = False
    fake_ctx.chunker.split.return_value = [fake_chunk]
    fake_ctx.embedder.embed_documents.return_value = [[0.1] * 768]

    with patch("src.streaming.consumer.MultimodalLoader") as mock_loader_class:
        mock_loader = MagicMock()
        mock_loader.load_single_document.return_value = fake_doc
        mock_loader_class.return_value = mock_loader

        from src.streaming.consumer import process_single_file

        process_single_file("/some/file.pdf", fake_ctx)

        fake_ctx.storage.save_embeddings.assert_called_once_with([fake_chunk], [[0.1] * 768])
