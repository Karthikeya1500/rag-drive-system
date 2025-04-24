"""
embedding/vector_store.py
Singleton SentenceTransformer, batch embeddings, FAISS index, disk persistence.
"""

import os
import pickle
import logging

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from config import settings

logger = logging.getLogger("documind.vector_store")

# ── Singleton model ───────────────────────────────────────────────────────────
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Load the embedding model once and cache it for the process lifetime."""
    global _model
    if _model is None:
        logger.info("Loading embedding model: %s", settings.EMBEDDING_MODEL)
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
        logger.info("Embedding model ready.")
    return _model


# ── Embedding creation ────────────────────────────────────────────────────────

def create_embeddings(chunks: list[dict], model: SentenceTransformer | None = None) -> np.ndarray:
    if model is None:
        model = get_model()
    texts = [c["text"] for c in chunks]
    logger.info("Encoding %d chunks…", len(texts))
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
    return np.array(embeddings, dtype=np.float32)


# ── FAISS index ───────────────────────────────────────────────────────────────

def build_faiss_index(embeddings: np.ndarray):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    logger.info("FAISS index built: %d vectors, dim=%d", len(embeddings), dimension)
    return index


# ── Disk persistence ──────────────────────────────────────────────────────────

def save_index(index, chunks: list[dict], data_dir: str | None = None) -> None:
    """Persist FAISS index and chunk list to disk so restarts are zero-cost."""
    data_dir = data_dir or settings.DATA_DIR
    os.makedirs(data_dir, exist_ok=True)

    faiss.write_index(index, os.path.join(data_dir, "faiss.index"))
    with open(os.path.join(data_dir, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)

    logger.info("Saved %d chunks to disk (%s).", len(chunks), data_dir)


def load_index(data_dir: str | None = None) -> tuple:
    """
    Load a previously persisted FAISS index + chunks from disk.

    Returns:
        (index, chunks) if found, else (None, [])
    """
    data_dir = data_dir or settings.DATA_DIR
    index_path  = os.path.join(data_dir, "faiss.index")
    chunks_path = os.path.join(data_dir, "chunks.pkl")

    if os.path.exists(index_path) and os.path.exists(chunks_path):
        index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)
        logger.info("Loaded persisted index: %d chunks from %s.", len(chunks), data_dir)
        return index, chunks

    logger.info("No persisted index found in %s.", data_dir)
    return None, []