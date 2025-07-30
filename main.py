
import os
import sys
import uuid
from typing import List, Optional, Union, Any
from datetime import datetime
import traceback


import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, String, Text, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import PyPDF2
import nltk
from nltk.tokenize import sent_tokenize

from chunking import recursive_chunking, sentence_chunking, custom_fixed_size_chunking
from text_extraction import extract_text_from_pdf, extract_text_from_txt
from pydantic_models import UploadResponse, SearchResponse, SearchResult, DocumentMetadataResponse

# Download NLTK punkt tokenizer for sentence splitting
try:
    nltk.data.find('tokenizers/punkt')
except Exception:
    nltk.download('punkt')


# Configuration
# Database for metadata
SQLITE_DATABASE_URL = "sqlite:///./metadata.db"

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION_NAME = "document_chunks"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

Base = declarative_base()

class DocumentMetadata(Base):
    """SQLAlchemy model for storing document metadata."""
    __tablename__ = "document_metadata"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    file_name = Column(String, nullable=False)
    original_file_name = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    chunking_method = Column(String, nullable=False)
    embedding_model = Column(String, nullable=False)
    upload_date = Column(DateTime, default=datetime.utcnow)
    chunks_info = Column(JSON) # Store info about chunks, e.g., chunk_id, start_char, end_char


engine = create_engine(SQLITE_DATABASE_URL, connect_args={"check_same_thread": False})
Base.metadata.create_all(bind=engine)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

try:
    qdrant_client.recreate_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
    )
    print(f"Qdrant collection '{QDRANT_COLLECTION_NAME}' created/recreated.")
except Exception as e:
    print(f"Failed to create/recreate Qdrant collection: {e}. It might already exist.")
    try:
        qdrant_client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
    except Exception as e_get:
        print(f"Could not get Qdrant collection either: {e_get}. Please ensure Qdrant is running and accessible.")
        sys.exit(1)


try:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print(f"Embedding model '{EMBEDDING_MODEL_NAME}' loaded successfully.")
except Exception as e:
    print(f"Error loading embedding model '{EMBEDDING_MODEL_NAME}': {e}")
    sys.exit(1)


# FastAPI Application
app = FastAPI(
    title="Document Processing Backend",
    description="API for uploading documents, chunking, embedding, and storing in vector database.",
    version="1.0.0",
)

@app.post("/upload-file/", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    chunking_method: str = Form(..., description="Chunking method: 'recursive', 'semantic' (sentence-based), or 'custom'"),
):
    """
    Uploads a document (.pdf or .txt), extracts text, chunks it,
    generates embeddings, and stores them in Qdrant.
    Metadata is saved in SQLite.
    """
    if file.content_type not in ["application/pdf", "text/plain"]:
        raise HTTPException(status_code=400, detail="Unsupported file type. Only .pdf and .txt are allowed.")

    file_extension = file.filename.split(".")[-1].lower()
    temp_file_path = f"temp_{uuid.uuid4()}.{file_extension}"

    # Save the uploaded file temporarily
    try:
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save file: {e}")

    extracted_text = ""
    if file_extension == "pdf":
        extracted_text = extract_text_from_pdf(temp_file_path)
    elif file_extension == "txt":
        extracted_text = extract_text_from_txt(temp_file_path)

    # Remove temporary file
    os.remove(temp_file_path)

    if not extracted_text:
        raise HTTPException(status_code=400, detail="Could not extract text from the document.")

   
    chunks = []
    if chunking_method == "recursive":
        chunks = recursive_chunking(extracted_text)
    elif chunking_method == "semantic": # Using sentence-based as a proxy for semantic
        chunks = sentence_chunking(extracted_text)
    elif chunking_method == "custom":
        chunks = custom_fixed_size_chunking(extracted_text)
    else:
        raise HTTPException(status_code=400, detail="Invalid chunking method. Choose 'recursive', 'semantic', or 'custom'.")

    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks generated from the document. Document might be empty or too small.")

    # Generate embeddings for chunks and prepare for Qdrant
    points = []
    chunks_info_for_metadata = []
    document_id = str(uuid.uuid4())

    for i, chunk in enumerate(chunks):
        chunk_id = str(uuid.uuid4())
        embedding = embedding_model.encode(chunk).tolist()
        points.append(
            models.PointStruct(
                id=chunk_id,
                vector=embedding,
                payload={
                    "document_id": document_id,
                    "chunk_text": chunk,
                    "chunk_index": i,
                    "file_name": file.filename,
                    "chunking_method": chunking_method,
                    "embedding_model": EMBEDDING_MODEL_NAME,
                },
            )
        )
        chunks_info_for_metadata.append({
            "chunk_id": chunk_id,
            "chunk_index": i,
            "length": len(chunk),
            "preview": chunk[:100] + "..." if len(chunk) > 100 else chunk
        })

    # Store embeddings in Qdrant
    try:
        qdrant_client.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            wait=True,
            points=points,
        )
        print(f"Uploaded {len(chunks)} chunks to Qdrant for document '{file.filename}'.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store embeddings in Qdrant: {e}")

    # Store metadata in SQLite
    db = SessionLocal()
    try:
        new_metadata = DocumentMetadata(
            id=document_id,
            file_name=file.filename,
            original_file_name=file.filename,
            file_type=file_extension,
            chunking_method=chunking_method,
            embedding_model=EMBEDDING_MODEL_NAME,
            chunks_info=chunks_info_for_metadata
        )
        db.add(new_metadata)
        db.commit()
        db.refresh(new_metadata)
        print(f"Metadata for document '{file.filename}' saved to SQLite.")
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save metadata to SQLite: {e}")
    finally:
        db.close()

    return UploadResponse(
        id=document_id,
        file_name=file.filename,
        chunking_method=chunking_method,
        embedding_model=EMBEDDING_MODEL_NAME,
        num_chunks=len(chunks),
        message="File uploaded, processed, and stored successfully."
    )

@app.get("/search/", response_model=SearchResponse)
async def search_documents(query: str, limit: int = 5):
    """
    Searches for relevant document chunks in Qdrant based on a query.
    """
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        query_embedding = embedding_model.encode(query).tolist()
        search_result = qdrant_client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=query_embedding,
            limit=limit,
            append_payload=True # To include payload i.e. chunk text, metadata in results
        )

        results = []
        for hit in search_result:
            results.append(
                SearchResult(
                    chunk_id=hit.id,
                    text=hit.payload.get("chunk_text", ""),
                    score=hit.score,
                    metadata={
                        "document_id": hit.payload.get("document_id"),
                        "file_name": hit.payload.get("file_name"),
                        "chunk_index": hit.payload.get("chunk_index"),
                        "chunking_method": hit.payload.get("chunking_method"),
                        "embedding_model": hit.payload.get("embedding_model"),
                    }
                )
            )
        return SearchResponse(query=query, results=results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during search: {e}")

@app.get("/documents/", response_model=List[DocumentMetadataResponse])
async def get_all_documents():
    """
    Retrieves all stored document metadata.
    """
    db = SessionLocal()
    try:
        documents = db.query(DocumentMetadata).all()
        return documents
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving documents: {e}")
    finally:
        db.close()
