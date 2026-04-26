"""
main.py — DocuMind AI RAG Drive System (Production)
"""

# Load env vars first, before any other import reads os.getenv()
from dotenv import load_dotenv
load_dotenv(override=True)

import os

# Limit PyTorch / BLAS threads to prevent deadlocks & GIL starvation on 1 CPU
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import logging
import threading
import urllib.request
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config import settings
from connectors.google_drive import sync_drive_files
from processing.document_processor import process_documents
from embedding.vector_store import (
    create_embeddings, build_faiss_index, get_model,
    save_index, load_index,
)
from search.search import search_chunks
from api.llm import generate_answer

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("documind")

# ── Global in-memory state ────────────────────────────────────────────────────
chunks_data: list[dict] = []
faiss_index = None
sync_status: dict = {
    "status": "idle",
    "message": "Not synced yet. Call POST /sync-drive to start.",
    "new_files": [], "skipped_files": [], "total_files": 0, "chunks": 0,
}


# ── Keep-alive: self-ping every 10 min to prevent Render cold starts ──────────
def _keep_alive():
    """Pings /health every 10 minutes so Render free tier never sleeps."""
    import time
    time.sleep(60)  # wait 1 min after boot before first ping
    while True:
        try:
            port = os.environ.get("PORT", "8000")
            urllib.request.urlopen(f"http://localhost:{port}/health", timeout=10)
            logger.debug("Keep-alive ping sent.")
        except Exception:
            pass
        time.sleep(600)  # 10 minutes


# ── Startup: load persisted index ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global chunks_data, faiss_index, sync_status
    # Pre-warm embedding model so first request is instant
    get_model()
    # Load previously persisted index (survives restarts)
    loaded_index, loaded_chunks = load_index(settings.DATA_DIR)
    if loaded_index is not None:
        faiss_index  = loaded_index
        chunks_data  = loaded_chunks
        sync_status  = {
            "status":        "ready",
            "message":       f"Loaded {len(chunks_data)} chunks from disk (previous sync).",
            "new_files":     [],
            "skipped_files": [],
            "total_files":   len({c["metadata"]["file_name"] for c in chunks_data}),
            "chunks":        len(chunks_data),
        }
        logger.info("Startup: loaded %d chunks from persisted index.", len(chunks_data))
    else:
        logger.info("Startup: no persisted index found. Call /sync-drive.")
    # Start keep-alive background thread
    t = threading.Thread(target=_keep_alive, daemon=True)
    t.start()
    yield
    logger.info("Shutdown: application stopped.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DocuMind AI",
    description="Personal ChatGPT over your Google Drive documents.",
    version="1.0.0",
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


# ── Authentication ────────────────────────────────────────────────────────────
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_auth(api_key: str | None = Security(_api_key_header)) -> None:
    """FastAPI dependency: validates X-API-Key if API_KEY is configured."""
    if settings.API_KEY and api_key != settings.API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key.")


# ── Request / Response models ─────────────────────────────────────────────────
class AskRequest(BaseModel):
    query: str
    file_filter: Optional[str] = None


class AskResponse(BaseModel):
    query: str
    answer: str
    sources: list[dict]


# ── Background sync pipeline (runs in a real daemon thread) ──────────────────
_sync_lock = threading.Lock()


def run_sync() -> None:
    global chunks_data, faiss_index, sync_status

    with _sync_lock:
        try:
            sync_status = {**sync_status, "status": "syncing",
                           "message": "Step 1/4 — Connecting to Google Drive…"}
            logger.info("Sync started.")

            downloaded, skipped = sync_drive_files()
            logger.info("Downloaded %d file(s), skipped %d.", len(downloaded), len(skipped))

            sync_status = {**sync_status,
                           "message": f"Step 2/4 — Downloaded {len(downloaded)} file(s). Extracting text…"}
            chunks_data = process_documents(settings.DOWNLOAD_DIR)

            sync_status = {**sync_status,
                           "message": f"Step 3/4 — Generating embeddings for {len(chunks_data)} chunks…"}
            if chunks_data:
                model       = get_model()
                embeddings  = create_embeddings(chunks_data, model)
                faiss_index = build_faiss_index(embeddings)
                sync_status = {**sync_status, "message": "Step 4/4 — Saving index to disk…"}
                save_index(faiss_index, chunks_data, settings.DATA_DIR)

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

@app.get("/api/info", summary="Public server info")
def api_info():
    """Returns whether API-key authentication is enabled."""
    return {"version": "1.0.0", "auth_required": bool(settings.API_KEY)}


@app.post("/api/verify-key", summary="Verify API key")
def verify_key(_: None = Depends(require_auth)):
    return {"valid": True}


@app.post("/sync-drive", summary="Trigger Google Drive sync",
          dependencies=[Depends(require_auth)])
def sync_drive():
    global sync_status
    if _sync_lock.locked():
        return {"message": "Sync already in progress. Check GET /sync-status."}
    sync_status = {**sync_status, "status": "syncing", "message": "Sync queued…"}
    t = threading.Thread(target=run_sync, daemon=True)
    t.start()
    return {"message": "Sync started in background. Poll GET /sync-status for progress."}


@app.get("/sync-status", summary="Check sync progress")
def get_sync_status():
    return sync_status


@app.post("/ask", response_model=AskResponse, summary="Ask a question",
          dependencies=[Depends(require_auth)])
def ask(request: AskRequest):
    global chunks_data, faiss_index

    if faiss_index is None or not chunks_data:
        return AskResponse(
            query=request.query,
            answer="The system is not ready. Call POST /sync-drive first.",
            sources=[],
        )

    model   = get_model()
    results = search_chunks(
        query=request.query,
        index=faiss_index,
        chunks=chunks_data,
        model=model,
        top_k=settings.TOP_K,
        file_filter=request.file_filter,
    )

    if not results:
        return AskResponse(
            query=request.query,
            answer="No relevant chunks found for your query.",
            sources=[],
        )

    try:
        answer = generate_answer(request.query, results)
    except Exception as exc:
        logger.error("LLM error: %s", exc)
        raise HTTPException(status_code=503,
                            detail="LLM temporarily unavailable. Please try again shortly.")

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

    logger.info("Query: %r → %d sources", request.query[:60], len(sources))
    return AskResponse(query=request.query, answer=answer, sources=sources)


@app.get("/health", summary="Health check")
def health():
    return {
        "status": "ok",
        "synced": faiss_index is not None,
        "chunks": len(chunks_data),
        "auth":   bool(settings.API_KEY),
    }


# ── Serve frontend (must be LAST) ─────────────────────────────────────────────
_frontend = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.isdir(_frontend):
    app.mount("/", StaticFiles(directory=_frontend, html=True), name="frontend")