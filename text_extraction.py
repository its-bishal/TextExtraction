from fastapi import HTTPException
import PyPDF2
import traceback

# --- Text Extraction Functions ---
def extract_text_from_pdf(file_path: str) -> str:
    """Extracts text from a PDF file."""
    text = ""
    try:
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text() or ""
    except Exception as e:
        # Log the full traceback for debugging
        print(f"Error extracting text from PDF: {e}")
        traceback.print_exc() # Print full traceback to console
        raise HTTPException(status_code=500, detail=f"Failed to extract text from PDF. Error: {e}. Check server logs for details.")
    return text

def extract_text_from_txt(file_path: str) -> str:
    """Extracts text from a TXT file."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading text file: {e}")
    return text