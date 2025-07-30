from pydantic import BaseModel
from typing import List
from datetime import datetime


# Pydantic model for response
class UploadResponse(BaseModel):
    id: str
    file_name: str
    chunking_method: str
    embedding_model: str
    num_chunks: int
    message: str

class SearchResult(BaseModel):
    chunk_id: str
    text: str
    score: float
    metadata: dict

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]

# Pydantic model for DocumentMetadata response
class DocumentMetadataResponse(BaseModel):
    id: str
    file_name: str
    original_file_name: str
    file_type: str
    chunking_method: str
    embedding_model: str
    upload_date: datetime
    chunks_info: List[dict] # Use List[dict] for the JSON column

    class Config:
        orm_mode = True # Enable ORM mode for Pydantic to read from SQLAlchemy models
