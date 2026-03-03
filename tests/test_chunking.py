import pytest
from src.ingestion.models import IngestedDocument
from src.processing.strategies.fixed import FixedSizeStrategy
from src.processing.strategies.sliding import SlidingWindowStrategy

def test_fixed_size_chunking():
    doc = IngestedDocument(filename="test.txt", content="A" * 150)
    strategy = FixedSizeStrategy(chunk_size=100, overlap=50)
    chunks = strategy.split([doc])
    
    assert len(chunks) == 2
    assert chunks[0].content == "A" * 100
    assert chunks[1].content == "A" * 100 # overlap means 50 to 150 = 100

def test_fixed_size_invalid_params():
    with pytest.raises(ValueError):
        FixedSizeStrategy(chunk_size=-10, overlap=5)
    
    with pytest.raises(ValueError):
        FixedSizeStrategy(chunk_size=100, overlap=100) # >= chunk_size

def test_sliding_window_chunking():
    doc = IngestedDocument(filename="test.txt", content="word " * 100)
    strategy = SlidingWindowStrategy(window_size=60, step_size=40)
    chunks = strategy.split([doc])
    
    assert len(chunks) == 2
    assert len(chunks[0].content.split()) == 60
    assert len(chunks[1].content.split()) == 60

def test_sliding_window_invalid_params():
    with pytest.raises(ValueError):
        SlidingWindowStrategy(window_size=0, step_size=10)
    
    with pytest.raises(ValueError):
        SlidingWindowStrategy(window_size=10, step_size=0)
