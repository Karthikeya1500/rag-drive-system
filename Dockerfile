# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.12-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python deps (cached layer) ────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application source ────────────────────────────────────────────────────
COPY . .

# ── Ensure storage directories and executable startup script ──────────────────
RUN mkdir -p downloads data && chmod +x startup.sh

EXPOSE 8000

CMD ["./startup.sh"]
