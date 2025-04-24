# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.12-slim

# System deps required by faiss-cpu and sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python deps (cached layer) ───────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Pre-download the embedding model ─────────────────────────────────────────
# This bakes the model into the image so cold starts are instant.
ARG EMBEDDING_MODEL=all-MiniLM-L6-v2
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')"

# ── Copy application source ───────────────────────────────────────────────────
COPY . .

# Ensure persistent storage directories exist
RUN mkdir -p downloads data

EXPOSE 8000

# ── Production server: gunicorn + uvicorn workers ────────────────────────────
# WORKERS env var controls parallelism (default 2; set higher on multi-core hosts)
CMD ["sh", "-c", "gunicorn main:app \
      -k uvicorn.workers.UvicornWorker \
      -w ${WORKERS:-2} \
      --bind 0.0.0.0:8000 \
      --timeout 120 \
      --access-logfile - \
      --error-logfile -"]
