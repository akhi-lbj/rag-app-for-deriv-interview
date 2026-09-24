import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from src.config import TOP_K, SIMILARITY_THRESHOLD, RETRIEVAL_RESULTS_FILE
from src.vector_store import FAISSVectorStore
from src.sparse_index import BM25Index
from src.ingestion import Chunk
from src.logger import logger

@dataclass
class RetrievedChunk:
    doc_id: str
    chunk_id: str
    score: float
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "score": round(self.score, 4),
            "text": self.text
        }

@dataclass
class QuestionRetrievalResult:
    question_id: str
    retrieved_chunks: List[RetrievedChunk]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "retrieved_chunks": [c.to_dict() for c in self.retrieved_chunks]
        }

class RetrievalPipeline:
    """
    Production Hybrid Retrieval Pipeline:
    Combines:
    1. Dense Vector Search (OpenAI embeddings + FAISS IndexFlatIP) for semantic comprehension.
    2. Sparse Lexical Search (BM25) for exact keyword/token/code matching.
    3. Reciprocal Rank Fusion (RRF) & Reranking to merge, score, and select top candidates.
    """
    def __init__(
        self,
        vector_store: FAISSVectorStore,
        sparse_index: Optional[BM25Index] = None,
        top_k: int = TOP_K,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        rrf_k: int = 60
    ):
        self.vector_store = vector_store
        self.sparse_index = sparse_index
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.rrf_k = rrf_k

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Tuple[Chunk, float]],
        sparse_results: List[Tuple[Chunk, float]]
    ) -> List[Tuple[Chunk, float]]:
        """
        Merges dense and sparse search rankings using Reciprocal Rank Fusion (RRF):
            Score(chunk) = sum( 1.0 / (rrf_k + rank_channel) )
        Plus score normalization for seamless downstream confidence gating.
        """
        chunk_map: Dict[str, Chunk] = {}
        rrf_scores: Dict[str, float] = {}
        dense_scores_map: Dict[str, float] = {}
        sparse_scores_map: Dict[str, float] = {}

        # Process dense ranking
        for rank, (chunk, score) in enumerate(dense_results):
            cid = chunk.chunk_id
            chunk_map[cid] = chunk
            dense_scores_map[cid] = score
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank + 1))

        # Process sparse (BM25) ranking
        for rank, (chunk, score) in enumerate(sparse_results):
            cid = chunk.chunk_id
            chunk_map[cid] = chunk
            sparse_scores_map[cid] = score
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank + 1))

        # Normalize sparse scores into [0, 1]
        max_sparse = max(sparse_scores_map.values()) if sparse_scores_map else 1.0
        if max_sparse <= 0.0:
            max_sparse = 1.0

        # Reranker scoring:
        # Fuses dense cosine similarity, normalized BM25 lexical match, and reciprocal rank agreement
        fused_candidates: List[Tuple[Chunk, float]] = []
        for cid, chunk in chunk_map.items():
            dense_s = dense_scores_map.get(cid, 0.0)
            norm_sparse_s = sparse_scores_map.get(cid, 0.0) / max_sparse
            
            # If present in both channels, apply cross-modal agreement boost
            agreement_boost = 1.15 if (cid in dense_scores_map and cid in sparse_scores_map) else 1.0
            
            # Composite hybrid score
            hybrid_score = (0.65 * dense_s + 0.35 * norm_sparse_s) * agreement_boost
            fused_candidates.append((chunk, float(hybrid_score)))

        # Sort descending by composite score
        fused_candidates.sort(key=lambda x: x[1], reverse=True)
        return fused_candidates

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        """
        Executes hybrid retrieval:
        1. Query dense FAISS index.
        2. Query sparse BM25 index.
        3. RRF + Rerank candidates.
        4. Deduplicate text chunks.
        """
        k = top_k or self.top_k
        fetch_k = max(k * 2, 6)

        # 1. Dense FAISS search
        dense_results = self.vector_store.search(query, top_k=fetch_k)

        # 2. Sparse BM25 search
        sparse_results: List[Tuple[Chunk, float]] = []
        if self.sparse_index is not None:
            sparse_results = self.sparse_index.search(query, top_k=fetch_k)

        # 3. Hybrid RRF & Reranking
        if sparse_results:
            ranked_results = self._reciprocal_rank_fusion(dense_results, sparse_results)
        else:
            ranked_results = dense_results

        # 4. Deduplicate near-identical chunk texts and apply threshold
        retrieved: List[RetrievedChunk] = []
        seen_texts = set()

        for chunk, score in ranked_results:
            norm_text = " ".join(chunk.text.split())
            if norm_text in seen_texts:
                continue
            seen_texts.add(norm_text)

            retrieved.append(
                RetrievedChunk(
                    doc_id=chunk.doc_id,
                    chunk_id=chunk.chunk_id,
                    score=float(score),
                    text=chunk.text
                )
            )
            if len(retrieved) >= k:
                break

        top_s = retrieved[0].score if retrieved else 0.0
        logger.info(
            f"[HYBRID RETRIEVAL] Query: '{query}' -> Fused {len(ranked_results)} candidates down to {len(retrieved)} chunks (Top hybrid score: {top_s:.3f})"
        )
        return retrieved

    def evaluate_questions(self, questions: List[Dict[str, Any]], output_file: Path = RETRIEVAL_RESULTS_FILE) -> List[QuestionRetrievalResult]:
        """
        Executes retrieval for a list of evaluation questions and persists retrieval_results.json.
        """
        results: List[QuestionRetrievalResult] = []
        logger.info(f"[RETRIEVAL] Running hybrid retrieval for {len(questions)} evaluation questions...")

        for q in questions:
            q_id = q.get("id", "unknown")
            query_text = q.get("question", "")
            chunks = self.retrieve(query_text)
            results.append(QuestionRetrievalResult(question_id=q_id, retrieved_chunks=chunks))

        # Save to retrieval_results.json
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)

        logger.info(f"[RETRIEVAL] Saved hybrid retrieval results to {output_file}")
        return results
