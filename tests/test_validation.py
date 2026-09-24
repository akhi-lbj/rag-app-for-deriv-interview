import pytest
from src.validation import CitationValidator

def test_valid_supported_answer():
    validator = CitationValidator()
    answer_record = {
        "question_id": "q1",
        "answer": "Users can reset passwords 3 times per hour.",
        "citations": ["account_security.md"],
        "supported": True
    }
    retrieved_chunks = [
        {"doc_id": "account_security.md", "chunk_id": "c1", "score": 0.9}
    ]
    result = validator.validate_record(answer_record, retrieved_chunks)
    assert result.is_valid is True
    assert result.checks["citations_present_if_supported"] is True
    assert result.checks["citations_in_retrieved_docs"] is True

def test_missing_citations_fails_supported_answer():
    validator = CitationValidator()
    answer_record = {
        "question_id": "q1",
        "answer": "Users can reset passwords 3 times per hour.",
        "citations": [],
        "supported": True
    }
    retrieved_chunks = [
        {"doc_id": "account_security.md", "chunk_id": "c1", "score": 0.9}
    ]
    result = validator.validate_record(answer_record, retrieved_chunks)
    assert result.is_valid is False
    assert result.checks["citations_present_if_supported"] is False

def test_hallucinated_citation_fails():
    validator = CitationValidator()
    answer_record = {
        "question_id": "q1",
        "answer": "Some policy statement.",
        "citations": ["non_existent_doc.md"],
        "supported": True
    }
    retrieved_chunks = [
        {"doc_id": "account_security.md", "chunk_id": "c1", "score": 0.9}
    ]
    result = validator.validate_record(answer_record, retrieved_chunks)
    assert result.is_valid is False
    assert result.checks["citations_in_retrieved_docs"] is False

def test_unsupported_refusal_passes():
    validator = CitationValidator()
    answer_record = {
        "question_id": "q6",
        "answer": "The retrieved documentation does not support this claim.",
        "citations": [],
        "supported": False
    }
    retrieved_chunks = [
        {"doc_id": "kyc_verification.md", "chunk_id": "c1", "score": 0.3}
    ]
    result = validator.validate_record(answer_record, retrieved_chunks)
    assert result.is_valid is True
    assert result.checks["supported_flag_consistent"] is True

def test_empty_answer_fails():
    validator = CitationValidator()
    answer_record = {
        "question_id": "q_empty",
        "answer": "   ",
        "citations": [],
        "supported": False
    }
    retrieved_chunks = []
    result = validator.validate_record(answer_record, retrieved_chunks)
    assert result.is_valid is False
    assert result.checks["non_empty_answer"] is False
