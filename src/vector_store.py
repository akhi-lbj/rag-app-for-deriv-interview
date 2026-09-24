import os
import pickle
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import faiss
from openai import OpenAI
from src.config import OPENAI_API_KEY, EMBEDDING_MODEL
from src.ingestion import Chunk
from src.logger import logger

class FAISSVectorStore:
    def __init__(self, dimension: int = 1536, model_name: str = EMBEDDING_MODEL):
        self.dimension = dimension
        self.model_name = model_name
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product for Cosine Similarity with normalized vectors
        self.chunks: List[Chunk] = []
        self._openai_client: Optional[OpenAI] = None
        if OPENAI_API_KEY:
            try:
                self._openai_client = OpenAI(api_key=OPENAI_API_KEY)
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}. Falling back to deterministic embedding.")
    def _hash_to_dim(self, term: str) -> int:
        return int(hashlib.md5(term.encode("utf-8")).hexdigest(), 16) % self.dimension

    def _fallback_embed(self, text: str) -> np.ndarray:
        """
        Deterministic local pseudo-semantic embedding fallback when OPENAI_API_KEY is not available.
        Uses character and subword n-gram feature hashing via MD5 to produce a normalized 1536-dim vector.
        Guarantees that pipeline runs seamlessly on clean checkout without an external API key,
        completely consistent across separate process runs.
        """
        vec = np.zeros(self.dimension, dtype=np.float32)
        words = text.lower().split()
        for i, word in enumerate(words):
            # Term hash
            h1 = self._hash_to_dim(word)
            vec[h1] += 1.0
            # Bigram hash
            if i > 0:
                bigram = f"{words[i-1]}_{word}"
                h2 = self._hash_to_dim(bigram)
                vec[h2] += 1.5
            # Character trigram hash
            for c_idx in range(len(word) - 2):
                tri = word[c_idx:c_idx+3]
                h3 = self._hash_to_dim(tri)
                vec[h3] += 0.5
        
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generates dense embeddings for a list of texts using OpenAI text-embedding-3-small,
        with automated fallback if API key is missing.
        """
        if self._openai_client is not None:
            try:
                # Clean strings for embedding
                cleaned_texts = [t.replace("\n", " ") for t in texts]
                batch_size = 64
                all_embeddings = []
                for i in range(0, len(cleaned_texts), batch_size):
                    batch = cleaned_texts[i:i + batch_size]
                    response = self._openai_client.embeddings.create(
                        input=batch,
                        model=self.model_name
                    )
                    batch_vecs = [item.embedding for item in response.data]
                    all_embeddings.extend(batch_vecs)
                
                vectors = np.array(all_embeddings, dtype=np.float32)
                faiss.normalize_L2(vectors)
                return vectors
            except Exception as e:
                logger.warning(f"[INDEXING] OpenAI embedding API call failed: {e}. Using deterministic local fallback.")

        # Local fallback
        vectors = np.array([self._fallback_embed(t) for t in texts], dtype=np.float32)
        faiss.normalize_L2(vectors)
        return vectors

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embeds a single query string.
        """
        return self.embed_texts([query])

    def build_index(self, chunks: List[Chunk]):
        """
        Indexes chunks into the FAISS vector database.
        """
        if not chunks:
            logger.warning("[INDEXING] No chunks provided to index.")
            return

        logger.info(f"[INDEXING] Generating embeddings for {len(chunks)} chunks using {self.model_name}...")
        texts = [chunk.text for chunk in chunks]
        vectors = self.embed_texts(texts)
        
        # Reset and add
        self.index.reset()
        self.index.add(vectors)
        self.chunks = list(chunks)
        logger.info(f"[INDEXING] Successfully built FAISS IndexFlatIP with {self.index.ntotal} vectors.")

    def search(self, query: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        """
        Performs vector similarity search against the FAISS index.
        Returns top_k (chunk, cosine_similarity_score) pairs.
        """
        if self.index.ntotal == 0 or not self.chunks:
            logger.warning("[RETRIEVAL] FAISS index is empty.")
            return []

        query_vec = self.embed_query(query)
        scores, indices = self.index.search(query_vec, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))

        return results

    def save(self, directory: Path):
        """
        Serializes FAISS index and chunk metadata to disk.
        """
        directory.mkdir(parents=True, exist_ok=True)
        index_file = directory / "index.faiss"
        meta_file = directory / "metadata.pkl"
        
        faiss.write_index(self.index, str(index_file))
        with open(meta_file, "wb") as f:
            pickle.dump(self.chunks, f)
        logger.info(f"[INDEXING] FAISS index and metadata saved to {directory}")

    def load(self, directory: Path) -> bool:
        """
        Loads FAISS index and chunk metadata from disk.
        """
        index_file = directory / "index.faiss"
        meta_file = directory / "metadata.pkl"
        if not index_file.exists() or not meta_file.exists():
            return False

        self.index = faiss.read_index(str(index_file))
        with open(meta_file, "rb") as f:
            self.chunks = pickle.load(f)
        logger.info(f"[INDEXING] Loaded FAISS index ({self.index.ntotal} entries) from {directory}")
        return True
