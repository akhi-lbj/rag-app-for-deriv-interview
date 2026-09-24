import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "docs"
QUESTIONS_FILE = BASE_DIR / "questions.json"
RETRIEVAL_RESULTS_FILE = BASE_DIR / "retrieval_results.json"
ANSWERS_FILE = BASE_DIR / "answers.json"
VALIDATION_REPORT_FILE = BASE_DIR / "validation_report.json"
LOG_FILE = BASE_DIR / "pipeline.log"

_raw_key = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_API_KEY = _raw_key if _raw_key and not _raw_key.startswith("your_openai_api_key") else ""
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gpt-4o-mini")

# Ingestion / Chunking configuration
CHUNK_SIZE = 500  # characters
CHUNK_OVERLAP = 100  # characters

# Retrieval configuration
TOP_K = 3
SIMILARITY_THRESHOLD = 0.20  # minimum similarity score to consider relevant
