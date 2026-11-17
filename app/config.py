"""Central configuration for the Agentic RAG system."""
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# Paths
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# LLM (Groq)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# Embeddings
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# Chunking
CHUNK_SIZE = 1000       # characters per chunk
CHUNK_OVERLAP = 200     # overlap between chunks

# Retrieval
TOP_K = 5

# Files
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
CHROMA_COLLECTION = "knowledge_base"



# Helper model for lightweight tasks.
GROQ_FAST_MODEL = os.getenv("GROQ_FAST_MODEL", GROQ_MODEL)

# Minimum cosine similarity score for relevant search results.
MIN_SCORE = 0.2
