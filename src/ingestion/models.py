from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime
import uuid

class IngestedDocument(BaseModel):
    """Represents a raw document ingested from the source."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    ingested_at: datetime = Field(default_factory=datetime.utcnow)

class ProcessedChunk(BaseModel):
    """Represents a chunk of text that has been processed and is ready for embedding."""
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parent_doc_id: str
    content: str
    chunk_index: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
