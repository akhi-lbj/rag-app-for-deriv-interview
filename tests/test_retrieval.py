import pytest
from src.ingestion import Chunk
from src.vector_store import FAISSVectorStore
from src.sparse_index import BM25Index
from src.retrieval import RetrievalPipeline

@pytest.fixture
def sample_chunks():
    return [
        Chunk(
            doc_id="account_security.md",
            chunk_id="account_security.md_chunk_0",
            chunk_index=0,
            text="Users are permitted a maximum of 3 password reset attempts per hour before lockout.",
            char_start=0,
            char_end=84,
            source_path="/fake/account_security.md"
        ),
        Chunk(
            doc_id="api_rate_limits.md",
            chunk_id="api_rate_limits.md_chunk_0",
            chunk_index=0,
            text="Standard tier is rate limited to 60 requests per minute and returns HTTP 429.",
            char_start=0,
            char_end=77,
            source_path="/fake/api_rate_limits.md"
        ),
        Chunk(
            doc_id="kyc_verification.md",
            chunk_id="kyc_verification.md_chunk_0",
            chunk_index=0,
            text="Tier 2 KYC proof of address verification requires 24 to 48 business hours.",
            char_start=0,
            char_end=74,
            source_path="/fake/kyc_verification.md"
        )
    ]

@pytest.fixture
def populated_vector_store(sample_chunks):
    store = FAISSVectorStore()
    store.build_index(sample_chunks)
    return store

@pytest.fixture
def populated_bm25_index(sample_chunks):
    sparse = BM25Index()
    sparse.build_index(sample_chunks)
    return sparse

def test_retrieval_returns_relevant_document(populated_vector_store):
    pipeline = RetrievalPipeline(vector_store=populated_vector_store, top_k=2)
    results = pipeline.retrieve("How many password reset attempts are allowed?")
    
    assert len(results) > 0
    top_doc_id = results[0].doc_id
    assert top_doc_id == "account_security.md"
    assert results[0].score > 0.0

def test_retrieval_returns_rate_limits(populated_vector_store):
    pipeline = RetrievalPipeline(vector_store=populated_vector_store, top_k=1)
    results = pipeline.retrieve("What happens when HTTP 429 rate limit is reached?")
    
    assert len(results) == 1
    assert results[0].doc_id == "api_rate_limits.md"

def test_bm25_sparse_search(populated_bm25_index):
    results = populated_bm25_index.search("HTTP 429 rate limit", top_k=2)
    assert len(results) > 0
    top_chunk, score = results[0]
    assert top_chunk.doc_id == "api_rate_limits.md"
    assert score > 0.0

def test_hybrid_retrieval_fusion(populated_vector_store, populated_bm25_index):
    pipeline = RetrievalPipeline(
        vector_store=populated_vector_store,
        sparse_index=populated_bm25_index,
        top_k=2
    )
    # Query with exact technical term and semantic description
    results = pipeline.retrieve("What status code HTTP 429 means in API rate limits?")
    assert len(results) > 0
    assert results[0].doc_id == "api_rate_limits.md"
    # Ensure score reflects composite hybrid reranking
    assert results[0].score > 0.0
