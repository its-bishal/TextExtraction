from langchain.text_splitter import RecursiveCharacterTextSplitter
from nltk.tokenize import sent_tokenize

from typing import List, Optional, Union, Any

# --- Chunking Strategies ---
def recursive_chunking(text: str) -> List[str]:
    """Chunks text using Langchain's RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_text(text)
    return chunks

def sentence_chunking(text: str) -> List[str]:
    """Chunks text by sentences, ensuring sentences are not split."""
    # Using NLTK's sentence tokenizer
    sentences = sent_tokenize(text)
    # Simple aggregation to form chunks, ensuring sentences stay together
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= 1000: # +1 for space
            current_chunk += (" " if current_chunk else "") + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def custom_fixed_size_chunking(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Custom fixed-size chunking with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)
        if start >= len(text) - overlap and end < len(text): # Handle last small chunk
            remaining_text = text[end:]
            if remaining_text:
                chunks.append(remaining_text)
            break # Exit after processing the last part
    return chunks