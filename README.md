# DocuMind AI — RAG Drive System

> **Your personal ChatGPT over Google Drive.** Upload documents to Google Drive, ask questions, and get grounded answers with source citations — powered by RAG (Retrieval-Augmented Generation).

---

## Architecture

```
rag-drive-system/
├── connectors/
│   └── google_drive.py       # Service Account auth, file listing, download, incremental sync
├── processing/
│   └── document_processor.py # PDF / DOCX / TXT extraction, paragraph-aware chunking, metadata
├── embedding/
│   └── vector_store.py       # SentenceTransformer (all-MiniLM-L6-v2), FAISS index builder
├── search/
│   └── search.py             # Semantic search with optional metadata filtering
├── api/
│   └── llm.py                # Groq / LLaMA-3.1 answer generation
├── frontend/
│   └── index.html            # Premium chat UI (served by FastAPI)
├── main.py                   # FastAPI app, background sync pipeline
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

### Data Flow

```
Google Drive
    │ (Service Account — share files with bot email)
    ▼
POST /sync-drive  →  Background Task
    ├── 1. Download new/changed files (incremental sync via manifest)
    ├── 2. Extract text  →  PDF | DOCX | TXT
    ├── 3. Clean & chunk (paragraph-aware, 600 chars, 120 overlap)
    ├── 4. Encode chunks → vectors  (all-MiniLM-L6-v2)
    └── 5. Build FAISS IndexFlatL2

POST /ask  {"query": "..."}
    ├── 1. Encode query → vector
    ├── 2. FAISS search → top-5 chunks (optional file_filter)
    ├── 3. LLM prompt (Groq LLaMA-3.1-8b) with labeled excerpts
    └── 4. Return answer + deduplicated sources
```

---

## Setup

### Prerequisites
- Python 3.12+
- A Google Cloud project with **Drive API** enabled
- A **Service Account** with a downloaded `credentials.json`
- A **Groq API key** (free at [console.groq.com](https://console.groq.com))

### 1 — Clone & Install

```bash
git clone https://github.com/your-username/rag-drive-system.git
cd rag-drive-system
pip install -r requirements.txt
```

### 2 — Configure Environment

Create a `.env` file:

```env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxx
```

Place your Google service account key as `credentials.json` in the project root.

### 3 — Share Google Drive Files with the Bot

In Google Drive, share documents with:
```
<your-service-account-email>@<project-id>.iam.gserviceaccount.com
```
Set permission to **Viewer**.

Supported file types: **PDF**, **Google Docs** (exported as TXT), **DOCX**, **TXT**

### 4 — Run

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

> Mount `credentials.json` and `.env` as volumes — they are never baked into the image.

---

## API Reference

### `POST /sync-drive`
Trigger an incremental sync of Google Drive documents (runs as a background task).

```bash
curl -X POST http://localhost:8000/sync-drive
```

**Response:**
```json
{"message": "Sync started in background. Poll GET /sync-status for progress."}
```

---

### `GET /sync-status`
Poll sync progress.

```bash
curl http://localhost:8000/sync-status
```

**Response (ready):**
```json
{
  "status": "ready",
  "message": "Sync complete. 3 new, 2 unchanged.",
  "new_files": ["policy.pdf", "sop.docx", "readme.txt"],
  "skipped_files": ["old_report.pdf", "data.txt"],
  "total_files": 5,
  "chunks": 142
}
```

---

### `POST /ask`
Ask a question grounded in your documents.

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is our refund policy?"}'
```

**Response:**
```json
{
  "query": "What is our refund policy?",
  "answer": "According to policy.pdf, customers are eligible for a full refund within 30 days of purchase. After 30 days, store credit is offered instead.",
  "sources": [
    {
      "doc_id": "a3f1c...",
      "file_name": "policy.pdf",
      "file_type": "pdf",
      "source": "gdrive",
      "page": 4
    }
  ]
}
```

**Optional metadata filtering** (search only within a specific file):
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the compliance rules?", "file_filter": "compliance"}'
```

---

### `GET /health`
```json
{"status": "ok", "synced": true, "chunks": 142}
```

---

## Sample Queries & Outputs

| Query | Expected Behaviour |
|---|---|
| `"What is our refund policy?"` | Extracts refund terms from policy docs |
| `"Summarize the key findings"` | Summarises report sections |
| `"What are the compliance requirements?"` | Pulls rules from compliance docs |
| `"What formulas are used in chapter 4?"` | Finds formulas from PDF page content |
| `"List the steps in the onboarding SOP"` | Ordered steps from SOP document |

---

## Evaluation Checklist

| Criterion | Status |
|---|---|
| ✅ Google Drive integration (Service Account) | Done |
| ✅ PDF extraction | Done |
| ✅ Google Docs & DOCX & TXT extraction | Done |
| ✅ Text cleaning & normalisation | Done |
| ✅ Paragraph-aware chunking with overlap | Done |
| ✅ Metadata: doc_id, file_name, source, page, chunk_index | Done |
| ✅ SentenceTransformers embeddings (batch) | Done |
| ✅ FAISS vector store | Done |
| ✅ `POST /sync-drive` | Done |
| ✅ `POST /ask` with JSON body | Done |
| ✅ Answer with sources returned | Done |
| ✅ Incremental sync (skips unchanged files) | Done |
| ✅ Async background pipeline + status polling | Done |
| ✅ Metadata filtering (`file_filter` param) | Done |
| ✅ Docker + Docker Compose | Done |
| ✅ Premium frontend UI | Done |

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| Drive Connector | Google Drive API v3 (Service Account) |
| Document Parsing | pypdf, python-docx |
| Embeddings | SentenceTransformers (`all-MiniLM-L6-v2`) |
| Vector Store | FAISS (IndexFlatL2) |
| LLM | Groq — LLaMA 3.1 8B Instant |
| Frontend | Vanilla HTML/CSS/JS (TailwindCSS CDN) |
| Containerisation | Docker + Docker Compose |
