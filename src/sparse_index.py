import math
import re
from collections import Counter
from typing import List, Tuple, Dict
from src.ingestion import Chunk

class BM25Index:
    """
    Okapi BM25 implementation for lexical keyword retrieval.
    Complements dense vector embeddings by precisely matching exact technical terms,
    error codes (e.g. 'HTTP 429'), acronyms ('KYC', 'TOTP', 'P1', 'PIR'), and numbers.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: List[Chunk] = []
        self.corpus_size = 0
        self.avg_doc_len = 0.0
        self.doc_lens: List[int] = []
        self.doc_freqs: Dict[str, int] = Counter()
        self.term_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}

    def _tokenize(self, text: str) -> List[str]:
        """Simple lowercase alphanumeric tokenizer preserving hyphenated terms."""
        return [t.lower() for t in re.findall(r"[a-zA-Z0-9_\-\$]+", text) if len(t) > 1]

    def build_index(self, chunks: List[Chunk]):
        """Builds BM25 term frequency tables and IDF dictionary from chunks."""
        self.chunks = list(chunks)
        self.corpus_size = len(chunks)
        if self.corpus_size == 0:
            return

        self.doc_lens = []
        self.term_freqs = []
        total_len = 0
        df = Counter()

        for chunk in chunks:
            tokens = self._tokenize(chunk.text)
            length = len(tokens)
            self.doc_lens.append(length)
            total_len += length
            
            tf = Counter(tokens)
            self.term_freqs.append(tf)
            for token in tf.keys():
                df[token] += 1

        self.avg_doc_len = total_len / self.corpus_size
        self.doc_freqs = df

        # Compute IDF with Robertson-Spärck Jones smoothing
        self.idf = {}
        for term, freq in df.items():
            # Standard BM25 IDF formulation
            self.idf[term] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """
        Calculates Okapi BM25 scores for query across indexed chunks.
        Returns top_k (chunk, bm25_score) sorted descending.
        """
        if self.corpus_size == 0:
            return []

        query_tokens = self._tokenize(query)
        scores: List[float] = [0.0] * self.corpus_size

        for term in query_tokens:
            if term not in self.idf:
                continue
            idf_val = self.idf[term]
            
            for idx in range(self.corpus_size):
                tf = self.term_freqs[idx].get(term, 0)
                if tf == 0:
                    continue
                doc_len = self.doc_lens[idx]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / (self.avg_doc_len or 1.0)))
                scores[idx] += idf_val * (numerator / denominator)

        # Rank and return top_k
        ranked_indices = sorted(range(self.corpus_size), key=lambda i: scores[i], reverse=True)
        results = []
        for idx in ranked_indices[:top_k]:
            if scores[idx] > 0.0:
                results.append((self.chunks[idx], float(scores[idx])))

        return results
