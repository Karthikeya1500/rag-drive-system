"""
search/bm25_search.py
Lightweight BM25 + TF-IDF search — no large model weights required.
Works within Render free tier (512 MB RAM) unlike sentence-transformers.
"""

import re
import math
import pickle
import os
import logging
from collections import Counter

logger = logging.getLogger("documind.bm25")


# ── Tokenizer ────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]


# ── BM25 Index ───────────────────────────────────────────────────────────────

class BM25Index:
    """
    BM25 Okapi implementation.
    k1=1.5 and b=0.75 are standard defaults.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: list[dict] = []
        self.tokenized: list[list[str]] = []
        self.df: dict[str, int] = {}   # document frequency per term
        self.idf: dict[str, float] = {}
        self.avgdl: float = 0.0
        self.N: int = 0

    def build(self, chunks: list[dict]) -> None:
        self.chunks = chunks
        self.tokenized = [_tokenize(c["text"]) for c in chunks]
        self.N = len(chunks)

        # Document frequency
        self.df = {}
        for tokens in self.tokenized:
            for term in set(tokens):
                self.df[term] = self.df.get(term, 0) + 1

        # IDF (with smoothing)
        self.idf = {
            term: math.log((self.N - df + 0.5) / (df + 0.5) + 1)
            for term, df in self.df.items()
        }

        # Average document length
        lengths = [len(t) for t in self.tokenized]
        self.avgdl = sum(lengths) / max(self.N, 1)

        logger.info("BM25 index built: %d chunks, avg_len=%.1f", self.N, self.avgdl)

    def search(self, query: str, top_k: int = 5,
               file_filter: str | None = None) -> list[dict]:
        """Return top_k chunks ranked by BM25 score."""
        if not self.chunks:
            return []

        q_terms = _tokenize(query)
        scores: list[float] = []

        for i, tokens in enumerate(self.tokenized):
            tf = Counter(tokens)
            dl = len(tokens)
            score = 0.0
            for term in q_terms:
                if term not in self.idf:
                    continue
                f = tf.get(term, 0)
                score += self.idf[term] * (
                    f * (self.k1 + 1) /
                    (f + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1)))
                )
            scores.append(score)

        # Pair scores with chunks; apply file filter
        ranked = sorted(
            (
                (scores[i], self.chunks[i])
                for i in range(len(self.chunks))
                if file_filter is None
                or self.chunks[i]["metadata"]["file_name"] == file_filter
            ),
            key=lambda x: x[0],
            reverse=True,
        )

        return [chunk for score, chunk in ranked[:top_k] if score > 0]


# ── Disk persistence ─────────────────────────────────────────────────────────

def save_bm25(index: BM25Index, data_dir: str) -> None:
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "bm25.pkl")
    with open(path, "wb") as f:
        pickle.dump(index, f)
    logger.info("BM25 index saved to %s (%d chunks).", path, index.N)


def load_bm25(data_dir: str) -> BM25Index | None:
    path = os.path.join(data_dir, "bm25.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            index = pickle.load(f)
        logger.info("BM25 index loaded from %s (%d chunks).", path, index.N)
        return index
    return None
