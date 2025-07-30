Document Processing Backend with FastAPI and QdrantThis project implements a RESTful API backend using FastAPI for processing documents.
It allows users to upload PDF or text files, extract their content, apply various text chunking strategies, generate embeddings, and store them in a vector database (Qdrant) for efficient similarity search.
All document metadata is persisted in a SQLite database.
FeaturesFile Uploads: Supports .pdf and .txt file uploads.
Text Extraction: Extracts plain text content from uploaded documents.

Flexible Chunking: Implements three chunking strategies:
Recursive Character Chunking: Uses langchain's RecursiveCharacterTextSplitter for intelligent splitting.
Semantic (Sentence-based) Chunking: Ensures sentences are kept intact.
Custom Fixed-Size Chunking: Simple fixed-size chunks with overlap.

Embedding Generation: Generates vector embeddings for text chunks using the all-MiniLM-L6-v2 Sentence Transformer model.
Vector Database Storage: Stores embeddings and associated metadata in Qdrant.Metadata Management: Persists document and chunk metadata (file name, chunking method, embedding model, etc.) in a SQLite database using SQLAlchemy.
Semantic Search: Allows searching for relevant document chunks based on a natural language query.

1. Clone the Repositorygit clone <your-repository-url>
cd <TextExtraction>
2. Create a Virtual Environment (Recommended)python -m venv venv
# On Windows
.\venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
3. Install DependenciesInstall all required Python libraries using the requirements.txt file:pip install -r requirements.txt
The application will also attempt to download the punkt tokenizer from NLTK if not found. If you encounter issues, you can manually download it:import nltk
nltk.download('punkt')
4. Run QdrantQdrant is used as the vector database. The easiest way to run it is via Docker:docker run -p 6333:6333 qdrant/qdrant or you could download and run the standalone executable from the Qdrant GitHub.
5. Run the FastAPI ApplicationOnce Qdrant is running, start the FastAPI application:uvicorn main:app --reload
The --reload flag will automatically restart the server on code changes.The API will be accessible at http://127.0.0.1:8000. API EndpointsThe API documentation (Swagger UI) is available at http://127.0.0.1:8000/docs.

1.Upload DocumentUploads a .pdf or .txt file, processes it, and stores chunks and metadata.Endpoint: POST /upload-file/Request Body (multipart/form-data):file: The document file (.pdf or .txt).
chunking_method: String, one of recursive, semantic, or custom.Example (using curl):
# For a text file
curl -X 'POST' \
  'http://127.0.0.1:8000/upload-file/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@path/to/your/document.txt;type=text/plain' \
  -F 'chunking_method=recursive'

# For a PDF file
curl -X 'POST' \
  'http://127.0.0.1:8000/upload-file/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@path/to/your/document.pdf;type=application/pdf' \
  -F 'chunking_method=semantic'

2. Search DocumentsSearches for relevant document chunks based on a query string.Endpoint: GET /search/Query Parameters:query: String, the search query.limit: Integer (optional, default: 5), maximum number of results to return.Example (using curl):curl -X 'GET' \
  'http://127.0.0.1:8000/search/?query=what+is+the+main+topic&limit=3' \
  -H 'accept: application/json'

3. Get All Document MetadataRetrieves metadata for all documents stored in the system.Endpoint: GET /documents/Example (using curl):curl -X 'GET' \
  'http://127.0.0.1:8000/documents/' \
  -H 'accept: application/json'
