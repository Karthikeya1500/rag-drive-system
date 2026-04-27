# DocuMind AI — RAG Drive System

> Your personal ChatGPT over Google Drive.
> Connect your Google Drive, sync documents, and get accurate AI-powered answers with source citations — powered by Retrieval-Augmented Generation (RAG).

**Live Demo:** https://rag-drive-system.onrender.com

---

## Architecture

```
rag-drive-system/
├── connectors/
│   └── google_drive.py       # Service Account auth, incremental sync, manifest tracking
├── processing/
│   └── document_processor.py # PDF / DOCX / TXT extraction, cleaning, paragraph-aware chunking
├── embedding/
│   └── vector_store.py       # Google gemini-embedding-001 API, FAISS index build & persist
├── search/
│   ├── search.py             # Semantic FAISS search with metadata filtering
│   └── bm25_search.py        # BM25 keyword search (fallback)
├── api/
│   └── llm.py                # Groq LLaMA-3.1 answer generation with retry logic
├── frontend/
│   └── index.html            # Chat UI served by FastAPI
├── main.py                   # FastAPI app, background sync pipeline, keep-alive
├── config.py                 # Centralised settings from environment
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

### Data Flow

```
Google Drive (shared files)
        |
        v
POST /sync-drive  ->  Background Thread
        |-- 1. List files via Drive API v3 (service account)
        |-- 2. Incremental sync: skip unchanged files (MD5/modifiedTime manifest)
        |-- 3. Download new/changed PDFs, Google Docs, DOCX, TXT
        |-- 4. Extract text (pypdf / python-docx / plain text)
        |-- 5. Clean and chunk (paragraph-aware, 600 chars, 120-char overlap)
        |-- 6. Embed chunks via Google gemini-embedding-001 (3072-dim vectors)
        |-- 7. Build FAISS IndexFlatL2 + BM25 keyword fallback index
        +-- 8. Persist both indexes to disk (survive container restarts)

POST /ask  {"query": "..."}
        |-- 1. Embed query via gemini-embedding-001 (RETRIEVAL_QUERY task type)
        |-- 2. FAISS semantic search -> top-5 most relevant chunks
        |         (optional file_filter for per-document queries)
        |-- 3. Build grounded prompt with labeled document excerpts
        |-- 4. Groq LLaMA-3.1-8b-instant generates answer
        +-- 5. Return {answer, sources} with deduplicated file citations
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn + Gunicorn |
| Drive Connector | Google Drive API v3 (Service Account) |
| Document Parsing | pypdf, python-docx |
| Embeddings | Google `gemini-embedding-001` (3072-dim, free API) |
| Vector Store | FAISS `IndexFlatL2` |
| Keyword Fallback | BM25 Okapi (pure Python) |
| LLM | Groq — LLaMA 3.1 8B Instant |
| Frontend | Vanilla HTML / CSS / JS |
| Containerisation | Docker + Docker Compose |
| Deployment | Render (Docker runtime) |

---

## Setup

### Prerequisites

- Python 3.12+
- Google Cloud project with **Drive API** enabled
- **Service Account** with `credentials.json` downloaded
- **Groq API key** (free at [console.groq.com](https://console.groq.com))
- **Google AI Studio API key** (free at [aistudio.google.com](https://aistudio.google.com))

### 1. Clone and Install

```bash
git clone https://github.com/Karthikeya1500/rag-drive-system.git
cd rag-drive-system
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the project root:

```env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
GOOGLE_API_KEY=AIzaSyxxxxxxxxxxxxxxxxx
```

Place your Google service account key as `credentials.json` in the project root.

### 3. Share Google Drive Files with the Bot

In Google Drive, share your documents with the service account email:

```
rag-drive-bot@<your-project-id>.iam.gserviceaccount.com
```

Set permission to **Viewer**. Supported file types: PDF, Google Docs, DOCX, TXT.

### 4. Run

```bash
uvicorn main:app --reload
```

Open **http://127.0.0.1:8000** in your browser.

---

## Running with Docker

```bash
# Build and start
docker-compose up --build

# Stop
docker-compose down
```

Set `GROQ_API_KEY` and `GOOGLE_API_KEY` in your environment or `.env` before running.

---

## API Reference

### `POST /sync-drive`

Trigger an incremental sync of Google Drive documents. Runs in a background thread — non-blocking.

```bash
curl -X POST http://localhost:8000/sync-drive
```

Response:

```json
{"message": "Sync started. Poll GET /sync-status for progress."}
```

---

### `GET /sync-status`

Poll sync progress in real-time.

```bash
curl http://localhost:8000/sync-status
```

Response while syncing:

```json
{
  "status": "syncing",
  "message": "Step 3/4 — Generating embeddings for 29 chunks"
}
```

Response when ready:

```json
{
  "status": "ready",
  "message": "Sync complete. 5 new, 0 unchanged.",
  "new_files": ["policy.pdf", "sop.docx"],
  "skipped_files": [],
  "total_files": 5,
  "chunks": 142
}
```

---

### `POST /ask`

Ask a question grounded in your synced documents.

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is our refund policy?"}'
```

Response:

```json
{
  "query": "What is our refund policy?",
  "answer": "According to policy.pdf (page 4), customers are eligible for a full refund within 30 days of purchase. After 30 days, only store credit is offered.",
  "sources": [
    {
      "doc_id": "a3f1c9d...",
      "file_name": "policy.pdf",
      "file_type": "pdf",
      "source": "gdrive",
      "page": 4
    }
  ]
}
```

Optional — filter search to a specific file:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "compliance rules", "file_filter": "compliance.pdf"}'
```

---

### `GET /files`

List all indexed documents and their chunk counts.

```bash
curl http://localhost:8000/files
```

Response:

```json
{
  "total_files": 5,
  "files": [
    {"file_name": "policy.pdf", "file_type": "pdf", "chunks": 18},
    {"file_name": "sop.docx", "file_type": "docx", "chunks": 12}
  ]
}
```

---

### `GET /health`

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "synced": true,
  "chunks": 142,
  "index": "faiss",
  "auth": false
}
```

---

## Sample Queries and Real Outputs

These are actual outputs from the deployed system with math exercise PDFs indexed.

**Query 1: Concept explanation**

```
Query:  "What is the quadratic formula?"
Answer: "The quadratic formula is x = (-b +/- sqrt(b^2 - 4ac)) / 2a.
         Your document '4.Quadratic Equations 2020.pdf' contains
         practice problems on this topic:
         1. Solve for x: 6x^2 + 11x + 3 = 0
         2. The quadratic equation x^2 - 4x + k = 0 has distinct
            real roots if (A) k=4 (B) k>4 (C) k=16 (D) k<4"
Sources: [4.Quadratic Equations 2020.pdf, page 1]
```

**Query 2: Document-specific question**

```
Query:  "What topics are covered in the arithmetic progressions document?"
Answer: "The document covers finding consecutive terms, common
         differences, sum of APs, and real-world AP problems.
         Example: find the 11th term from the last of AP 12, 8, 4, ..., -84."
Sources: [5.Arithmetic Progressions 2020.pdf, pages 1-3]
```

**Query 3: Metadata filtering**

```
Query:  "What geometry problems are there?"
        file_filter: "6.Coordinate Geometry 2020.pdf"
Answer: "According to 6.Coordinate Geometry 2020.pdf, problems include
         finding the area of triangles using coordinates and proving
         collinearity of points."
Sources: [6.Coordinate Geometry 2020.pdf, page 1]
```

---

## Evaluation Checklist

| Criterion | Status | Detail |
|---|---|---|
| Google Drive integration | Done | Service Account, Drive API v3 |
| Fetch PDF / Google Docs / TXT | Done | All 4 formats supported |
| `POST /sync-drive` | Done | Background thread, non-blocking |
| Text extraction | Done | pypdf + python-docx |
| Text cleaning and normalisation | Done | Whitespace, encoding normalisation |
| Meaningful chunking | Done | Paragraph-aware, 600 chars, 120 overlap |
| Metadata attached | Done | doc_id, file_name, source, page, chunk_index |
| Embedding layer | Done | Google gemini-embedding-001 (3072-dim) |
| Batch processing | Done | Batches of 10, rate-limit aware |
| FAISS vector store | Done | IndexFlatL2, persisted to disk |
| `POST /ask` | Done | Full RAG pipeline |
| Answer and sources returned | Done | Deduplicated source list with page numbers |
| Incremental sync | Exceptional | MD5/modifiedTime manifest, skips unchanged |
| Caching / persistence | Exceptional | Index survives container restarts |
| Metadata filtering | Exceptional | `file_filter` param on `/ask` |
| Async pipeline | Exceptional | Daemon thread + `/sync-status` polling |
| BM25 keyword fallback | Bonus | Keyword search when FAISS unavailable |
| Docker + Docker Compose | Bonus | Production-ready container |
| Deployed version | Bonus | https://rag-drive-system.onrender.com |
| Frontend UI | Bonus | Chat interface served by FastAPI |
