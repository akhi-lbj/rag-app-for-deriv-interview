import pytest
from src.ingestion import split_text_into_chunks, Chunk

def test_split_text_preserves_metadata():
    text = (
        "Line 1: Password reset policy allows 3 attempts.\n\n"
        "Line 2: After exceeding attempts, lockout is 60 minutes.\n\n"
        "Line 3: 2FA is required for all administrative actions."
    )
    doc_id = "test_doc.md"
    chunks = split_text_into_chunks(doc_id, text, "/path/test_doc.md", chunk_size=80, chunk_overlap=20)
    
    assert len(chunks) > 0
    for idx, chunk in enumerate(chunks):
        assert isinstance(chunk, Chunk)
        assert chunk.doc_id == doc_id
        assert chunk.chunk_id == f"{doc_id}_chunk_{idx}"
        assert chunk.chunk_index == idx
        assert chunk.char_start >= 0
        assert chunk.char_end > chunk.char_start
        assert len(chunk.text) > 0

def test_chunking_handles_empty_text():
    chunks = split_text_into_chunks("empty.md", "", "/path/empty.md")
    assert chunks == []
