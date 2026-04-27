"""
main.py — DocuMind AI RAG Drive System (Production)
Architecture: Google Drive → chunking → Google text-embedding-004 → FAISS → Groq LLM
"""

from dotenv import load_dotenv
load_dotenv(override=True)

import os
import logging
import threading
import urllib.request
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from config import settings
from connectors.google_drive import sync_drive_files
from processing.document_processor import process_documents
from embedding.vector_store import (
    create_embeddings, build_faiss_index,
    save_index, load_index,
)
from search.search import search_chunks
from search.bm25_search import BM25Index, save_bm25, load_bm25
from api.llm import generate_answer

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("documind")

# ── Global state ──────────────────────────────────────────────────────────────
chunks_data: list[dict] = []
faiss_index = None       # primary: semantic search
bm25_index: BM25Index | None = None   # fallback: keyword search

sync_status: dict = {
    "status":  "idle",
    "message": "Not synced yet. Click 'Sync Drive' to start.",
    "new_files": [], "skipped_files": [], "total_files": 0, "chunks": 0,
}


# ── Keep-alive ping ───────────────────────────────────────────────────────────
def _keep_alive() -> None:
    import time
    time.sleep(60)
    while True:
        try:
            port = os.environ.get("PORT", "8000")
            urllib.request.urlopen(f"http://localhost:{port}/health", timeout=10)
            logger.debug("Keep-alive ping sent.")
        except Exception:
            pass
        time.sleep(600)


# ── Startup ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global chunks_data, faiss_index, bm25_index, sync_status

    # Try semantic (FAISS) index first, then BM25 fallback
    loaded_idx, loaded_chunks = load_index(settings.DATA_DIR)
    if loaded_idx is not None:
        faiss_index = loaded_idx
        chunks_data = loaded_chunks
        bm25_index  = _build_bm25(chunks_data)
        sync_status = {
            "status":        "ready",
            "message":       f"Loaded {len(chunks_data)} chunks from disk.",
            "new_files":     [],
            "skipped_files": [],
            "total_files":   len({c["metadata"]["file_name"] for c in chunks_data}),
            "chunks":        len(chunks_data),
        }
        logger.info("Startup: loaded %d chunks (FAISS + BM25).", len(chunks_data))
    else:
        # Try BM25 fallback
        bm25 = load_bm25(settings.DATA_DIR)
        if bm25:
            bm25_index  = bm25
            chunks_data = bm25.chunks
            sync_status = {
                "status":        "ready",
                "message":       f"Loaded {len(chunks_data)} chunks (BM25 index).",
                "new_files":     [],
                "skipped_files": [],
                "total_files":   len({c["metadata"]["file_name"] for c in chunks_data}),
                "chunks":        len(chunks_data),
            }
            logger.info("Startup: loaded %d chunks from BM25 fallback.", len(chunks_data))
        else:
            logger.info("Startup: no persisted index found.")

    threading.Thread(target=_keep_alive, daemon=True).start()
    yield
    logger.info("Shutdown complete.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DocuMind AI",
    description="RAG system over Google Drive — powered by Google Embeddings + Groq LLM.",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Auth ──────────────────────────────────────────────────────────────────────
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_auth(api_key: str | None = Security(_api_key_header)) -> None:
    if settings.API_KEY and api_key != settings.API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key.")


# ── Models ────────────────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    query: str
    file_filter: Optional[str] = None


class AskResponse(BaseModel):
    query:   str
    answer:  str
    sources: list[dict]


# ── Helpers ───────────────────────────────────────────────────────────────────
def _build_bm25(chunks: list[dict]) -> BM25Index:
    idx = BM25Index()
    idx.build(chunks)
    return idx


# ── Sync pipeline ─────────────────────────────────────────────────────────────
_sync_lock = threading.Lock()


def run_sync() -> None:
    global chunks_data, faiss_index, bm25_index, sync_status

    with _sync_lock:
        try:
            # Step 1 — Google Drive
            sync_status = {**sync_status, "status": "syncing",
                           "message": "Step 1/4 — Connecting to Google Drive…"}
            logger.info("Sync started.")
            downloaded, skipped = sync_drive_files()
            logger.info("Drive: %d downloaded, %d skipped.", len(downloaded), len(skipped))

            # Step 2 — Parse & chunk
            sync_status = {**sync_status,
                           "message": f"Step 2/4 — Parsing {len(downloaded)} new file(s)…"}
            chunks_data = process_documents(settings.DOWNLOAD_DIR)
            logger.info("Processed %d chunks.", len(chunks_data))

            if not chunks_data:
                sync_status = {
                    "status": "ready", "message": "No documents found in Drive.",
                    "new_files": downloaded, "skipped_files": skipped,
                    "total_files": len(downloaded) + len(skipped), "chunks": 0,
                }
                return

            # Step 3 — Embed with Google text-embedding-004
            sync_status = {**sync_status,
                           "message": f"Step 3/4 — Generating embeddings for {len(chunks_data)} chunks…"}
            embeddings  = create_embeddings(chunks_data)
            faiss_index = build_faiss_index(embeddings)
            save_index(faiss_index, chunks_data, settings.DATA_DIR)

            # Step 4 — Build BM25 fallback index
            sync_status = {**sync_status, "message": "Step 4/4 — Building keyword index…"}
            bm25_index  = _build_bm25(chunks_data)
            save_bm25(bm25_index, settings.DATA_DIR)

            sync_status = {
                "status":        "ready",
                "message":       f"Sync complete. {len(downloaded)} new, {len(skipped)} unchanged.",
                "new_files":     downloaded,
                "skipped_files": skipped,
                "total_files":   len(downloaded) + len(skipped),
                "chunks":        len(chunks_data),
            }
            logger.info("Sync complete. %d chunks indexed.", len(chunks_data))

        except Exception as exc:
            logger.exception("Sync failed: %s", exc)
            sync_status = {**sync_status, "status": "error", "message": str(exc)}


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/api/info", summary="Server info")
def api_info():
    return {
        "version":      "2.0.0",
        "auth_required": bool(settings.API_KEY),
        "embedding":    "google/text-embedding-004",
        "llm":          settings.GROQ_MODEL,
        "search":       "FAISS (semantic) + BM25 (keyword fallback)",
    }


@app.post("/api/verify-key", summary="Verify API key")
def verify_key(_: None = Depends(require_auth)):
    return {"valid": True}


@app.post("/sync-drive", summary="Trigger Google Drive sync",
          dependencies=[Depends(require_auth)])
def sync_drive_route():
    global sync_status
    if _sync_lock.locked():
        return {"message": "Sync already in progress."}
    sync_status = {**sync_status, "status": "syncing", "message": "Sync queued…"}
    threading.Thread(target=run_sync, daemon=True).start()
    return {"message": "Sync started. Poll GET /sync-status for progress."}


@app.get("/sync-status", summary="Sync progress")
def get_sync_status():
    return sync_status


@app.get("/files", summary="List indexed files")
def list_files():
    files = {}
    for c in chunks_data:
        m = c["metadata"]
        fn = m["file_name"]
        if fn not in files:
            files[fn] = {"file_name": fn, "file_type": m["file_type"], "chunks": 0}
        files[fn]["chunks"] += 1
    return {"total_files": len(files), "files": list(files.values())}


@app.post("/ask", response_model=AskResponse, summary="Ask a question",
          dependencies=[Depends(require_auth)])
def ask(request: AskRequest):
    global chunks_data, faiss_index, bm25_index

    if not chunks_data:
        return AskResponse(
            query=request.query,
            answer="System not ready. Please click 'Sync Drive' first.",
            sources=[],
        )

    # Use FAISS (semantic) if available, else BM25 fallback
    if faiss_index is not None:
        results = search_chunks(
            query=request.query,
            index=faiss_index,
            chunks=chunks_data,
            top_k=settings.TOP_K,
            file_filter=request.file_filter,
        )
    elif bm25_index is not None:
        results = bm25_index.search(
            query=request.query,
            top_k=settings.TOP_K,
            file_filter=request.file_filter,
        )
    else:
        results = []

    if not results:
        return AskResponse(
            query=request.query,
            answer="No relevant content found for your query.",
            sources=[],
        )

    try:
        answer = generate_answer(request.query, results)
    except Exception as exc:
        logger.error("LLM error: %s", exc)
        raise HTTPException(status_code=503, detail="LLM temporarily unavailable.")

    seen: set[str] = set()
    sources: list[dict] = []
    for chunk in results:
        meta = chunk["metadata"]
        if meta["doc_id"] not in seen:
            seen.add(meta["doc_id"])
            sources.append({
                "doc_id":    meta["doc_id"],
                "file_name": meta["file_name"],
                "file_type": meta["file_type"],
                "source":    meta["source"],
                "page":      meta.get("page"),
            })

    logger.info("Q: %r → %d sources", request.query[:60], len(sources))
    return AskResponse(query=request.query, answer=answer, sources=sources)


@app.get("/health", summary="Health check")
def health():
    return {
        "status":  "ok",
        "synced":  faiss_index is not None or bm25_index is not None,
        "chunks":  len(chunks_data),
        "index":   "faiss" if faiss_index is not None else ("bm25" if bm25_index else "none"),
        "auth":    bool(settings.API_KEY),
    }


# ── Serve frontend ────────────────────────────────────────────────────────────
_frontend = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.isdir(_frontend):
    app.mount("/", StaticFiles(directory=_frontend, html=True), name="frontend")