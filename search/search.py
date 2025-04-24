"""
search/search.py
Semantic search over the FAISS index with optional metadata filtering.
"""

import numpy as np


def search_chunks(
    query: str,
    index,
    chunks: list[dict],
    model,
    top_k: int = 5,
    file_filter: str | None = None,
) -> list[dict]:
    """
    Retrieve the most relevant chunks for *query*.

    Args:
        query:       Natural-language question.
        index:       FAISS IndexFlatL2 object.
        chunks:      List of chunk dicts (text + metadata).
        model:       SentenceTransformer model for encoding the query.
        top_k:       Number of results to return.
        file_filter: Optional substring; restrict results to chunks whose
                     file_name contains this string (case-insensitive).

    Returns:
        Ordered list of chunk dicts (most relevant first).
    """
    if not chunks or index is None:
        return []

    query_vec = model.encode([query])
    query_vec = np.array(query_vec, dtype=np.float32)

    # Over-fetch when filtering so we still get top_k after dropping non-matches
    fetch_k = min(top_k * 6 if file_filter else top_k, len(chunks))
    distances, indices = index.search(query_vec, fetch_k)

    results: list[dict] = []
    for idx in indices[0]:
        if idx < 0 or idx >= len(chunks):
            continue

        chunk = chunks[idx]

        if file_filter:
            if file_filter.lower() not in chunk["metadata"]["file_name"].lower():
                continue

        results.append(chunk)
        if len(results) >= top_k:
            break

    return results
