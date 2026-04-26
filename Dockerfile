# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.12-slim

# System deps for faiss-cpu and sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python deps (cached layer) ───────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Pre-download embedding model (baked into image = instant cold start) ──────
ENV HF_HOME=/app/.cache/huggingface
ARG EMBEDDING_MODEL=all-MiniLM-L6-v2
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')"
ENV HF_HUB_OFFLINE=1
# ── Copy application source ───────────────────────────────────────────────────
COPY . .

# ── Ensure storage directories and executable startup script ─────────────────
RUN mkdir -p downloads data && chmod +x startup.sh

EXPOSE 8000

# Use startup.sh so credentials can be injected via env var on cloud platforms
CMD ["./startup.sh"]
