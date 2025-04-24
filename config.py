"""
config.py — Centralised settings loaded from environment variables.
Import this anywhere in the project; dotenv is loaded automatically.
"""

from dotenv import load_dotenv
load_dotenv(override=True)

import os
from typing import Optional


class Settings:
    # ── Server ────────────────────────────────────────────────────────────────
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    WORKERS: int = int(os.getenv("WORKERS", "2"))

    # ── Security ──────────────────────────────────────────────────────────────
    # Set API_KEY to enable authentication. Leave blank/unset for open access.
    API_KEY: Optional[str] = os.getenv("API_KEY") or None

    # Comma-separated allowed CORS origins. Set to your domain in production.
    ALLOWED_ORIGINS: list[str] = [
        o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()
    ]

    # ── Paths ─────────────────────────────────────────────────────────────────
    SERVICE_ACCOUNT_FILE: str = os.getenv("SERVICE_ACCOUNT_FILE", "credentials.json")
    DOWNLOAD_DIR: str  = os.getenv("DOWNLOAD_DIR", "downloads")
    DATA_DIR: str      = os.getenv("DATA_DIR", "data")   # persisted FAISS index

    # ── LLM ───────────────────────────────────────────────────────────────────
    GROQ_MODEL: str       = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    GROQ_MAX_TOKENS: int  = int(os.getenv("GROQ_MAX_TOKENS", "1024"))
    GROQ_TEMPERATURE: float = float(os.getenv("GROQ_TEMPERATURE", "0.1"))
    GROQ_MAX_RETRIES: int = int(os.getenv("GROQ_MAX_RETRIES", "3"))

    # ── Embedding ─────────────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # ── Search ────────────────────────────────────────────────────────────────
    TOP_K: int = int(os.getenv("TOP_K", "5"))


settings = Settings()
