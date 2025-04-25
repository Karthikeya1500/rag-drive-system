#!/bin/bash
set -e

echo "=== DocuMind AI — Starting up ==="

# ── Write Google credentials from environment variable ────────────────────────
# On cloud platforms, credentials.json is passed as a base64 env var
# instead of a file (since the file can't be committed to git).
if [ -n "$GOOGLE_CREDENTIALS_JSON" ]; then
    echo "Writing credentials.json from environment variable..."
    echo "$GOOGLE_CREDENTIALS_JSON" | base64 -d > /app/credentials.json
    echo "credentials.json written successfully."
elif [ ! -f /app/credentials.json ]; then
    echo "WARNING: credentials.json not found and GOOGLE_CREDENTIALS_JSON not set."
    echo "Google Drive sync will fail. Please set the GOOGLE_CREDENTIALS_JSON env var."
fi

# ── Ensure storage directories exist ─────────────────────────────────────────
mkdir -p /app/downloads /app/data

echo "=== Starting Gunicorn server ==="

# ── Start production server ───────────────────────────────────────────────────
# PORT is automatically injected by Render/Railway/Fly.io
exec gunicorn main:app \
    -k uvicorn.workers.UvicornWorker \
    -w "${WORKERS:-1}" \
    --bind "0.0.0.0:${PORT:-8000}" \
    --timeout 120 \
    --access-logfile - \
    --error-logfile -
