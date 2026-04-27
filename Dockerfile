# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.12-slim

# System deps for faiss-cpu
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python deps ───────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application source ───────────────────────────────────────────────────
COPY . .

# ── Ensure storage directories ────────────────────────────────────────────────
RUN mkdir -p downloads data && chmod +x startup.sh

EXPOSE 8000

CMD ["./startup.sh"]
