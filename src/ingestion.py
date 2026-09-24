from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any
import pypdf
from src.config import DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from src.logger import logger

@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    chunk_index: int
    text: str
    char_start: int
    char_end: int
    source_path: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

def read_text_file(file_path: Path) -> str:
    """Reads a text or markdown file."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()

def read_pdf_file(file_path: Path) -> str:
    """Extracts text content from a PDF file using pypdf."""
    reader = pypdf.PdfReader(str(file_path))
    pages_text = []
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages_text.append(f"[Page {page_num + 1}]\n{text}")
    return "\n\n".join(pages_text)

def load_documents(docs_dir: Path = DOCS_DIR) -> Dict[str, str]:
    """
    Loads all documents (.md, .txt, .pdf) from the specified directory.
    Returns mapping of doc_id -> full text.
    """
    documents = {}
    if not docs_dir.exists():
        logger.warning(f"Docs directory {docs_dir} does not exist.")
        return documents

    supported_extensions = {".md", ".txt", ".pdf"}
    for file_path in sorted(docs_dir.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            doc_id = file_path.name
            try:
                if file_path.suffix.lower() == ".pdf":
                    text = read_pdf_file(file_path)
                else:
                    text = read_text_file(file_path)
                
                if text.strip():
                    documents[doc_id] = text
                    logger.info(f"[INGESTION] Loaded document '{doc_id}' ({len(text)} chars)")
            except Exception as e:
                logger.error(f"[INGESTION] Failed to load {doc_id}: {e}")

    logger.info(f"[INGESTION] Total documents loaded: {len(documents)}")
    return documents

def split_text_into_chunks(doc_id: str, text: str, source_path: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[Chunk]:
    """
    Splits text into sliding window chunks with overlap, ensuring chunk metadata is preserved.
    Performs boundary normalization (e.g. attempting to break near paragraphs or sentences).
    """
    chunks = []
    step = chunk_size - chunk_overlap
    if step <= 0:
        step = chunk_size

    n = len(text)
    start = 0
    chunk_index = 0

    while start < n:
        end = min(start + chunk_size, n)
        
        # If we are not at the end of the text, try to find a natural break (newline or period)
        if end < n:
            break_pos = text.rfind("\n\n", start, end)
            if break_pos == -1 or break_pos < start + (chunk_size // 2):
                break_pos = text.rfind("\n", start, end)
            if break_pos == -1 or break_pos < start + (chunk_size // 2):
                break_pos = text.rfind(". ", start, end)
            
            if break_pos != -1 and break_pos > start + (chunk_size // 2):
                end = break_pos + 1

        chunk_str = text[start:end].strip()
        if chunk_str:
            chunk = Chunk(
                doc_id=doc_id,
                chunk_id=f"{doc_id}_chunk_{chunk_index}",
                chunk_index=chunk_index,
                text=chunk_str,
                char_start=start,
                char_end=end,
                source_path=source_path
            )
            chunks.append(chunk)
            chunk_index += 1

        if end >= n:
            break
        start += step

    return chunks

def ingest_corpus(docs_dir: Path = DOCS_DIR) -> List[Chunk]:
    """
    Full ingestion pipeline: loads all documents and chunks them.
    """
    documents = load_documents(docs_dir)
    all_chunks = []
    
    for doc_id, text in documents.items():
        doc_path = str(docs_dir / doc_id)
        chunks = split_text_into_chunks(doc_id, text, doc_path)
        all_chunks.extend(chunks)

    logger.info(f"[INGESTION] Total chunks created across all documents: {len(all_chunks)}")
    return all_chunks
