"""
embedding/vector_store.py
Google Gemini Embeddings (gemini-embedding-001) + FAISS vector store.
Uses Google's free embedding API — zero RAM, high quality 768-dim vectors.
"""

import os
import pickle
import logging
import time
from typing import Optional

import numpy as np
import faiss
import google.genai as genai

from config import settings

logger = logging.getLogger("documind.vector_store")

EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM   = 3072
_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.GOOGLE_API_KEY)
    return _client


# ── Embed a single text ───────────────────────────────────────────────────────
def _embed_text(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
    client = _get_client()
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config={"task_type": task_type},
    )
    return result.embeddings[0].values


# ── Batch embed chunks ────────────────────────────────────────────────────────
def create_embeddings(chunks: list[dict]) -> np.ndarray:
    """
    Embed all chunk texts using Google gemini-embedding-001.
    Free tier: 1,500 req/min — batches of 10 with small delay.
    Returns float32 ndarray of shape (N, 768).
    """
    texts = [c["text"] for c in chunks]
    logger.info("Embedding %d chunks via %s…", len(texts), EMBEDDING_MODEL)

    all_embeddings: list[list[float]] = []
    batch_size = 10

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        for text in batch:
            emb = _embed_text(text, task_type="RETRIEVAL_DOCUMENT")
            all_embeddings.append(emb)
        logger.info("  Embedded %d / %d chunks.", min(i + batch_size, len(texts)), len(texts))
        if i + batch_size < len(texts):
            time.sleep(0.5)   # stay well within free-tier rate limits

    arr = np.array(all_embeddings, dtype=np.float32)
    logger.info("Embeddings ready — shape: %s", arr.shape)
    return arr


def embed_query(query: str) -> np.ndarray:
    """Embed a user query for semantic retrieval."""
    emb = _embed_text(query, task_type="RETRIEVAL_QUERY")
    return np.array([emb], dtype=np.float32)


# ── FAISS index ───────────────────────────────────────────────────────────────
def build_faiss_index(embeddings: np.ndarray):
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    logger.info("FAISS index built: %d vectors, dim=%d", index.ntotal, dim)
    return index


# ── Disk persistence ──────────────────────────────────────────────────────────
def save_index(index, chunks: list[dict], data_dir: str) -> None:
    os.makedirs(data_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(data_dir, "faiss.index"))
    with open(os.path.join(data_dir, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)
    logger.info("Saved FAISS index + %d chunks → %s", len(chunks), data_dir)


def load_index(data_dir: str) -> tuple:
    """Returns (faiss_index, chunks) or (None, []) if not found."""
    idx_p    = os.path.join(data_dir, "faiss.index")
    chunk_p  = os.path.join(data_dir, "chunks.pkl")
    if os.path.exists(idx_p) and os.path.exists(chunk_p):
        index = faiss.read_index(idx_p)
        with open(chunk_p, "rb") as f:
            chunks = pickle.load(f)
        logger.info("Loaded FAISS index: %d chunks from %s", len(chunks), data_dir)
        return index, chunks
    logger.info("No persisted FAISS index in %s", data_dir)
    return None, []