import pytest
from src.ingestion.models import ProcessedChunk
from src.storage.manager import ParquetStorageManager

def test_manager_mismatch_length():
    chunk1 = ProcessedChunk(parent_doc_id="doc1", content="chunk 1", chunk_index=0)
    chunk2 = ProcessedChunk(parent_doc_id="doc1", content="chunk 2", chunk_index=1)
    
    embeddings = [[0.1, 0.2]] # Only 1 embedding for 2 chunks

    manager = ParquetStorageManager()
    
    with pytest.raises(ValueError) as excinfo:
        manager.save_embeddings([chunk1, chunk2], embeddings)
    
    assert "Mismatch: 2 chunks vs 1 embeddings" in str(excinfo.value)
