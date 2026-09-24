import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from src.config import BASE_DIR, DOCS_DIR
from src.ingestion import ingest_corpus
from src.vector_store import FAISSVectorStore
from src.retrieval import RetrievalPipeline
from src.generation import AnswerGenerator
from src.validation import CitationValidator
from src.logger import logger

# Initialize pipeline components
vector_store = FAISSVectorStore()
vector_store_dir = BASE_DIR / "vector_store"

def get_or_build_vector_store() -> FAISSVectorStore:
    """Loads existing index if available, else builds from docs/."""
    if not vector_store.load(vector_store_dir):
        logger.info("[APP] Vector store not found on disk. Building from docs/...")
        chunks = ingest_corpus(DOCS_DIR)
        vector_store.build_index(chunks)
        vector_store.save(vector_store_dir)
    return vector_store

retrieval_pipeline: Optional[RetrievalPipeline] = None
generator: Optional[AnswerGenerator] = None
validator: Optional[CitationValidator] = None

from src.sparse_index import BM25Index

def init_services():
    global retrieval_pipeline, generator, validator
    store = get_or_build_vector_store()
    
    sparse_index = BM25Index()
    if store.chunks:
        sparse_index.build_index(store.chunks)
    else:
        chunks = ingest_corpus(DOCS_DIR)
        sparse_index.build_index(chunks)

    retrieval_pipeline = RetrievalPipeline(vector_store=store, sparse_index=sparse_index)
    generator = AnswerGenerator()
    validator = CitationValidator()

def answer_query(question: str, top_k: int = 3) -> Dict[str, Any]:
    """Core answering function used by both CLI and API."""
    if retrieval_pipeline is None or generator is None or validator is None:
        init_services()

    # 1. Retrieve
    chunks = retrieval_pipeline.retrieve(question, top_k=top_k)

    # 2. Generate
    ans_record = generator.generate_answer(
        question_id="interactive",
        question=question,
        chunks=chunks
    )

    # 3. Validate
    chunk_dicts = [c.to_dict() for c in chunks]
    validation_res = validator.validate_record(ans_record.to_dict(), chunk_dicts)

    return {
        "question": question,
        "answer": ans_record.answer,
        "citations": ans_record.citations,
        "supported": ans_record.supported,
        "retrieved_sources": [
            {
                "doc_id": c.doc_id,
                "chunk_id": c.chunk_id,
                "score": round(c.score, 4)
            }
            for c in chunks
        ],
        "validation": {
            "is_valid": validation_res.is_valid,
            "checks": validation_res.checks,
            "details": validation_res.details
        }
    }

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_services()
    yield

# FastAPI App
app = FastAPI(
    title="AI Support Knowledge Base Assistant",
    description="Production-minded grounded question answering with citations and guardrails",
    version="1.0.0",
    lifespan=lifespan
)

class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3

class SourceInfo(BaseModel):
    doc_id: str
    chunk_id: str
    score: float

class ValidationInfo(BaseModel):
    is_valid: bool
    checks: Dict[str, bool]
    details: List[str]

class AskResponse(BaseModel):
    question: str
    answer: str
    citations: List[str]
    supported: bool
    retrieved_sources: List[SourceInfo]
    validation: ValidationInfo

from fastapi.responses import HTMLResponse, Response

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)

@app.get("/", response_class=HTMLResponse)
def root_page():
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>AI Support Knowledge Base</title>
  <style>
    :root {
      --bg: #0f172a;
      --card: #1e293b;
      --text: #f8fafc;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --success: #22c55e;
      --danger: #ef4444;
      --border: #334155;
    }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      max-width: 800px;
      margin: 40px auto;
      padding: 0 20px;
      line-height: 1.6;
    }
    h1 { color: #fff; margin-bottom: 8px; }
    p.desc { color: var(--muted); margin-top: 0; margin-bottom: 24px; }
    .nav { margin-bottom: 24px; font-size: 14px; }
    .nav a { color: var(--accent); text-decoration: none; margin-right: 16px; font-weight: 500; }
    .nav a:hover { text-decoration: underline; }
    .input-box { display: flex; gap: 10px; margin-bottom: 24px; }
    input[type="text"] {
      flex: 1;
      padding: 12px 16px;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: var(--card);
      color: #fff;
      font-size: 16px;
    }
    input[type="text"]:focus { outline: none; border-color: var(--accent); }
    button {
      padding: 12px 24px;
      background: var(--accent);
      color: #0f172a;
      border: none;
      border-radius: 8px;
      font-weight: 600;
      cursor: pointer;
      font-size: 16px;
    }
    button:hover { opacity: 0.9; }
    .samples { margin-bottom: 24px; font-size: 13px; color: var(--muted); }
    .sample-btn {
      display: inline-block;
      background: var(--card);
      color: var(--accent);
      padding: 4px 10px;
      border-radius: 6px;
      margin: 4px 4px 4px 0;
      cursor: pointer;
      border: 1px solid var(--border);
    }
    .sample-btn:hover { background: #293548; }
    .result {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 20px;
      display: none;
    }
    .badge {
      display: inline-block;
      padding: 4px 10px;
      border-radius: 12px;
      font-size: 12px;
      font-weight: bold;
      margin-bottom: 12px;
    }
    .badge-true { background: rgba(34, 197, 94, 0.2); color: var(--success); }
    .badge-false { background: rgba(239, 68, 68, 0.2); color: var(--danger); }
    .answer { font-size: 17px; margin-bottom: 16px; color: #fff; }
    .citations { margin-bottom: 16px; font-size: 14px; }
    .citations span { background: #334155; padding: 3px 8px; border-radius: 4px; margin-right: 6px; color: #38bdf8; }
    .sources { font-size: 13px; color: var(--muted); border-top: 1px solid var(--border); padding-top: 12px; }
  </style>
</head>
<body>
  <h1>AI Support Assistant</h1>
  <p class="desc">Grounded internal platform knowledge base with citations and deterministic guardrails.</p>
  
  <div class="nav">
    <a href="/docs" target="_blank">Swagger API Docs (OpenAPI)</a>
    <a href="/health" target="_blank">Health Check (/health)</a>
  </div>

  <div class="input-box">
    <input type="text" id="query" placeholder="Ask a support question..." value="How many password reset attempts are allowed per hour?" />
    <button onclick="askQuestion()">Ask</button>
  </div>

  <div class="samples">
    <strong>Sample Queries:</strong><br/>
    <span class="sample-btn" onclick="setQuery(this.innerText)">How many password reset attempts are allowed per hour?</span>
    <span class="sample-btn" onclick="setQuery(this.innerText)">What HTTP status code indicates API rate limit exceeded?</span>
    <span class="sample-btn" onclick="setQuery(this.innerText)">How long does standard Tier 2 identity verification take?</span>
    <span class="sample-btn" onclick="setQuery(this.innerText)">Can support agents manually bypass KYC verification for VIP clients?</span>
    <span class="sample-btn" onclick="setQuery(this.innerText)">What is the company dress code on casual Fridays?</span>
  </div>

  <div class="result" id="resultCard">
    <div id="badge"></div>
    <div class="answer" id="answerText"></div>
    <div class="citations" id="citationsBlock"></div>
    <div class="sources" id="sourcesBlock"></div>
  </div>

  <script>
    function setQuery(text) {
      document.getElementById('query').value = text;
      askQuestion();
    }
    async function askQuestion() {
      const q = document.getElementById('query').value.trim();
      if (!q) return;
      const resCard = document.getElementById('resultCard');
      resCard.style.display = 'block';
      document.getElementById('answerText').innerText = 'Searching knowledge base and generating answer...';
      document.getElementById('badge').innerHTML = '';
      document.getElementById('citationsBlock').innerHTML = '';
      document.getElementById('sourcesBlock').innerHTML = '';

      try {
        const resp = await fetch('/ask', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({question: q, top_k: 3})
        });
        const data = await resp.json();
        
        const badge = document.getElementById('badge');
        badge.innerHTML = data.supported 
          ? '<span class="badge badge-true">GROUNDED & SUPPORTED</span>' 
          : '<span class="badge badge-false">REFUSED / UNSUPPORTED</span>';

        document.getElementById('answerText').innerText = data.answer;

        const cit = document.getElementById('citationsBlock');
        if (data.citations && data.citations.length > 0) {
          cit.innerHTML = '<strong>Citations:</strong> ' + data.citations.map(c => `<span>${c}</span>`).join(' ');
        } else {
          cit.innerHTML = '<strong>Citations:</strong> <em>None (unsupported claim)</em>';
        }

        const src = document.getElementById('sourcesBlock');
        if (data.retrieved_sources) {
          src.innerHTML = '<strong>Retrieved Sources:</strong><br/>' + data.retrieved_sources.map(s => 
            `&bull; ${s.doc_id} (${s.chunk_id}, score: ${s.score})`
          ).join('<br/>');
        }
      } catch (err) {
        document.getElementById('answerText').innerText = 'Error querying API: ' + err.message;
      }
    }
  </script>
</body>
</html>
"""

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "ai-support-kb"}

@app.post("/ask", response_model=AskResponse)
def ask_endpoint(payload: AskRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    result = answer_query(payload.question, top_k=payload.top_k or 3)
    return result

def main():
    parser = argparse.ArgumentParser(description="AI Support Knowledge Base Service CLI & Server")
    parser.add_argument("--question", type=str, help="Ask a question directly via CLI")
    parser.add_argument("--serve", action="store_true", help="Start the FastAPI REST server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="API host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="API port (default: 8000)")
    args = parser.parse_args()

    if args.question:
        result = answer_query(args.question)
        print(json.dumps(result, indent=2))
    elif args.serve:
        print(f"Starting API server on http://{args.host}:{args.port} ...")
        uvicorn.run("app:app", host=args.host, port=args.port, reload=False)
    else:
        # Default behavior: print help or start CLI prompt
        parser.print_help()

if __name__ == "__main__":
    main()
