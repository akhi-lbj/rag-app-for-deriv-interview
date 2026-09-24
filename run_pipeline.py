import json
import sys
from pathlib import Path
from src.config import (
    DOCS_DIR,
    QUESTIONS_FILE,
    RETRIEVAL_RESULTS_FILE,
    ANSWERS_FILE,
    VALIDATION_REPORT_FILE,
    BASE_DIR
)
from src.ingestion import ingest_corpus
from src.vector_store import FAISSVectorStore
from src.retrieval import RetrievalPipeline
from src.generation import AnswerGenerator
from src.validation import CitationValidator
from src.logger import logger

def run_full_pipeline():
    logger.info("==================================================")
    logger.info("STARTING AI SUPPORT KNOWLEDGE BASE PIPELINE")
    logger.info("==================================================")

    # 1. Ingestion
    logger.info("[STEP 1/5] Ingesting documents from docs/...")
    chunks = ingest_corpus(DOCS_DIR)
    if not chunks:
        logger.error("No documents or chunks found in docs/ directory.")
        sys.exit(1)

    # 2. Hybrid Indexing (FAISS Dense Vector + BM25 Sparse Lexical)
    logger.info("[STEP 2/5] Indexing chunks in FAISS Vector Store and BM25 Sparse Index...")
    vector_store = FAISSVectorStore()
    vector_store.build_index(chunks)
    vector_store_dir = BASE_DIR / "vector_store"
    vector_store.save(vector_store_dir)

    from src.sparse_index import BM25Index
    sparse_index = BM25Index()
    sparse_index.build_index(chunks)

    # 3. Hybrid Retrieval with Reranking
    logger.info("[STEP 3/5] Loading questions and executing hybrid retrieval (Dense + Sparse RRF)...")
    if not QUESTIONS_FILE.exists():
        logger.error(f"Questions file {QUESTIONS_FILE} does not exist.")
        sys.exit(1)

    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    retrieval_pipeline = RetrievalPipeline(vector_store=vector_store, sparse_index=sparse_index)
    retrieval_results = retrieval_pipeline.evaluate_questions(questions, output_file=RETRIEVAL_RESULTS_FILE)

    # 4. Answer Generation with Grounding
    logger.info("[STEP 4/5] Generating grounded answers with citations...")
    generator = AnswerGenerator()
    
    # Pair questions with retrieval chunks
    questions_map = {q["id"]: q["question"] for q in questions}
    batch_input = [
        {
            "question_id": r.question_id,
            "question": questions_map.get(r.question_id, ""),
            "retrieved_chunks": [c.to_dict() for c in r.retrieved_chunks]
        }
        for r in retrieval_results
    ]
    answers = generator.generate_batch(batch_input, output_file=ANSWERS_FILE)

    # 5. Deterministic Validation
    logger.info("[STEP 5/5] Running deterministic citation and grounding checks...")
    validator = CitationValidator()
    
    with open(ANSWERS_FILE, "r", encoding="utf-8") as f:
        answers_data = json.load(f)
    with open(RETRIEVAL_RESULTS_FILE, "r", encoding="utf-8") as f:
        retrieval_data = json.load(f)

    report = validator.validate_all(answers_data, retrieval_data, output_file=VALIDATION_REPORT_FILE)

    # Print summary
    print("\n" + "="*60)
    print("PIPELINE EXECUTION SUMMARY")
    print("="*60)
    print(f"Total Questions Evaluated: {report['summary']['total_evaluated']}")
    print(f"Validation Passed:         {report['summary']['passed']}")
    print(f"Validation Failed:         {report['summary']['failed']}")
    print(f"Validation Pass Rate:      {report['summary']['pass_rate']}%")
    print("="*60)
    for res in report["results"]:
        q_id = res["question_id"]
        status = "PASSED" if res["is_valid"] else "FAILED"
        print(f"[{status}] Question ID: {q_id} | Checks: {res['checks']}")
    print("="*60 + "\n")
    logger.info("Pipeline completed successfully.")

if __name__ == "__main__":
    run_full_pipeline()
