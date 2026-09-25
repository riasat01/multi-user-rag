import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR/"data"/"uploads"
CHROMA_DIR = BASE_DIR/"data"/"chroma"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

EMBEDDING_MODEL = "models/gemini-embedding-001"
LLM_MODEL = "gemini-3.5-flash"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 4
COLLECTION_NAME = "documents"

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}