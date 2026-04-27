"""
search/search.py
FAISS semantic search using Google text-embedding-004 query embeddings.
"""

import logging
import numpy as np

logger = logging.getLogger("documind.search")


def search_chunks(
    query: str,
    index,
    chunks: list[dict],
    top_k: int = 5,
    file_filter: str | None = None,
) -> list[dict]:
    """
    Embed the query and retrieve top-K most similar chunks from FAISS.
    Optionally filter by file name.
    """
    from embedding.vector_store import embed_query

    query_vec = embed_query(query)
    k = min(top_k * 3, len(chunks))   # over-fetch to allow for filtering
    distances, indices = index.search(query_vec, k)

    results: list[dict] = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue
        chunk = chunks[idx]
        if file_filter and chunk["metadata"]["file_name"] != file_filter:
            continue
        results.append({**chunk, "score": float(dist)})
        if len(results) >= top_k:
            break

    logger.info("Search '%s' → %d results", query[:50], len(results))
    return results
