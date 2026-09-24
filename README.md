# AI Support Knowledge Base Assistant

A lightweight, runnable AI service that answers support questions grounded in local documentation, featuring hybrid retrieval, citations, and deterministic validation guardrails.

---

## 1. Architecture & File Layout

### Architecture Flow
```
docs/*.md ──> Ingestion & Chunking ──> Hybrid Indexing (FAISS Dense + BM25 Sparse)
                                                     │
User Question ───────────────────────────────────────┴──> Retrieval & RRF Reranking
                                                                     │
                                                              Top Chunks
                                                                     │
Answers + Citations <── Deterministic Validation <── Grounded Generation (LLM)
```

- **Ingestion & Indexing**: Markdown files are chunked with overlap and indexed into FAISS (OpenAI `text-embedding-3-small`) and an Okapi BM25 index.
- **Retrieval & Reranking**: Combines dense vector similarity and sparse keyword matching using Reciprocal Rank Fusion (RRF).
- **Generation & Guardrails**: Generates concise answers citing source documents, refuses unsupported claims, and deterministically validates citations.

### File Layout
```
├── docs/                      # 5 realistic policy & platform support docs
├── prompts/qa_prompt.py       # Isolated prompt templates
├── src/
│   ├── config.py              # Configuration & thresholds
│   ├── ingestion.py           # Document loading and sliding-window chunking
│   ├── vector_store.py        # FAISS vector store & OpenAI embeddings
│   ├── sparse_index.py        # Okapi BM25 keyword index
│   ├── retrieval.py           # Hybrid retrieval & RRF reranking
│   ├── generation.py          # Grounded answer synthesizer
│   ├── validation.py          # Deterministic citation and support validator
│   └── logger.py              # Structured logging
├── tests/                     # Unit tests (chunking, retrieval, validation)
├── questions.json             # 8 evaluation questions (5 answerable, 3 unanswerable)
├── retrieval_results.json     # Saved retrieval passages
├── answers.json               # Generated answers & citations
├── validation_report.json     # Deterministic validation report
├── run_pipeline.py            # End-to-end evaluation runner
├── app.py                     # Interactive CLI and FastAPI server
├── .env                       # Environment configuration
└── README.md
```

---

## 2. Setup Steps

### 1. Install Dependencies
Using `uv`:
```bash
uv sync
```
Or using standard `pip`:
```bash
pip install -r requirements.txt
```

### 2. Configure API Key
Set your OpenAI key in `.env`:
```ini
OPENAI_API_KEY=sk-...
EMBEDDING_MODEL=text-embedding-3-small
GENERATION_MODEL=gpt-4o-mini
```
*(Note: If `OPENAI_API_KEY` is not provided, the pipeline automatically falls back to an offline deterministic feature-hash embedding and extractive synthesizer so evaluation never crashes).*

### 3. Run the Pipeline
Run the full evaluation flow (ingests docs, indexes, retrieves, generates answers, validates, and regenerates all JSON artifacts):
```bash
python run_pipeline.py
```
*(or `uv run python run_pipeline.py` if using uv)*

### 4. Interactive CLI & API

#### Start the Server
Start the local API and interactive web interface:
```bash
python app.py --serve --port 8000
# or: uv run python app.py --serve --port 8000
```
- **Web UI**: Open `http://127.0.0.1:8000` in your browser for a visual question-answering playground.
- **API Docs**: Open `http://127.0.0.1:8000/docs` for interactive Swagger/OpenAPI documentation.

#### Run CLI Queries
Query the service directly from your terminal:
```bash
# Sample Answerable Question:
python app.py --question "How many password reset attempts are allowed per hour?"

# Sample Technical Code Question (matches via BM25):
python app.py --question "What HTTP status code and response headers indicate an API rate limit has been exceeded?"

# Sample Unsupported Question (triggers refusal):
python app.py --question "What is the company dress code on casual Fridays?"
```

#### Run Tests
```bash
pytest -v
# or: uv run pytest -v
```

---

## 3. How Retrieval Works (Hybrid Retrieval + Reranking)

The system implements **Hybrid Retrieval with Reciprocal Rank Fusion (RRF)**:

1. **Dense Vector Search (FAISS)**:
   - Chunks are embedded with OpenAI `text-embedding-3-small` (1536 dim) and $L_2$-normalized.
   - `faiss.IndexFlatIP` computes exact cosine similarity to capture semantic intent and synonyms.
2. **Sparse Lexical Search (BM25)**:
   - An Okapi BM25 index indexes exact tokens and handles technical terms, error codes (`HTTP 429`), and acronyms (`KYC`, `P1`, `SLA`).
3. **Fusion & Reranking**:
   - Candidates from both channels are merged using Reciprocal Rank Fusion:
     $$\text{RRF}(d) = \sum \frac{1}{60 + \text{rank}(d)}$$
   - Scores are blended ($0.65 \times \text{dense} + 0.35 \times \text{sparse}$) with a 15% agreement bonus for chunks matched by both channels, followed by near-duplicate filtering.

### Why BM25 & Reranking Were Chosen (Stretch Improvement)
- **Dense Embeddings Blind Spots**: Vector embeddings capture general meaning well, but frequently dilute exact alphanumeric strings, error codes (`HTTP 429`), protocol names (`TOTP`, `SEPA`), or specific incident codes (`P1`, `PIR`).
- **BM25 Lexical Precision**: BM25 directly scores exact keyword frequencies and token rarity, making it ideal for technical documentation with exact terminology.
- **RRF & Composite Reranking**: Merging them through Reciprocal Rank Fusion avoids the pitfall of incompatible score distributions, while our cross-modal reranker gives a confidence boost to passages confirmed by *both* channels.

---

## 4. How Grounding & Refusal Work

1. **Prompt Isolation**: System instructions in `prompts/qa_prompt.py` restrict the LLM to answer *only* from the provided context and format outputs as JSON with explicit citations.
2. **Confidence Thresholding**: If the top retrieval score is below `SIMILARITY_THRESHOLD` ($0.20$), the query is immediately rejected as unsupported without invoking the LLM.
3. **Structured Refusal Contract**: For unsupported questions, the model returns `supported: false` and empty citations `citations: []`.
4. **Deterministic Validation Gate** (`src/validation.py`):
   - Confirms the answer is non-empty.
   - Ensures supported answers include at least one citation.
   - Verifies all cited document IDs actually exist in the retrieved chunk set.
   - Ensures unsupported answers do not claim citations.

---

## 5. Limitations & Tradeoffs

- **Fixed-Window Chunking**: Character-based chunking with overlap is simple and fast, but can occasionally divide sentences across boundaries. Hierarchy-aware Markdown parsing would improve section boundaries.
- **Reranker Complexity**: A lightweight mathematical RRF reranker was chosen over a heavy Cross-Encoder model (e.g., `bge-reranker`) to keep dependencies minimal, fast, and CPU-friendly.
- **Deterministic Fallback vs. LLM**: The offline fallback provides resilience when an API key is missing by extracting sentences via keyword matching, but lacks the natural summarization capabilities of the full LLM.
