"""
connectors/google_drive.py
Google Drive sync connector with:
  - cache_discovery=False to skip the hanging discovery document fetch
  - httplib2 timeout so network hangs fail fast instead of blocking forever
  - Reads credentials from file OR from GOOGLE_CREDENTIALS_JSON env var (base64)
"""

import io
import os
import json
import base64
import tempfile
import logging

import httplib2  # kept for transport timeout in google-auth
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, HttpRequest

from config import settings

logger = logging.getLogger("documind.gdrive")

# ── Config ────────────────────────────────────────────────────────────────────
SCOPES       = ["https://www.googleapis.com/auth/drive.readonly"]
DOWNLOAD_DIR = settings.DOWNLOAD_DIR
MANIFEST_FILE = os.path.join(DOWNLOAD_DIR, ".manifest.json")

# MIME types the system supports and their local extensions
SUPPORTED_MIME = {
    "application/pdf": ".pdf",
    "application/vnd.google-apps.document": ".txt",   # exported as plain text
    "text/plain": ".txt",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
}

# ── Credentials ───────────────────────────────────────────────────────────────
def _get_credentials_path() -> str:
    """
    Returns a path to credentials.json.
    On cloud platforms, reads from GOOGLE_CREDENTIALS_JSON (base64-encoded).
    Falls back to the local credentials.json file.
    """
    b64 = os.environ.get("GOOGLE_CREDENTIALS_JSON", "").strip()
    if b64:
        try:
            data = base64.b64decode(b64)
            # Write to a temp file so google-auth can read it
            tmp = tempfile.NamedTemporaryFile(
                suffix=".json", delete=False, mode="wb"
            )
            tmp.write(data)
            tmp.close()
            logger.info("Credentials loaded from GOOGLE_CREDENTIALS_JSON env var.")
            return tmp.name
        except Exception as e:
            logger.warning("Failed to decode GOOGLE_CREDENTIALS_JSON: %s", e)

    # Fall back to file on disk (local dev or baked into Docker image)
    path = settings.SERVICE_ACCOUNT_FILE
    if os.path.exists(path):
        logger.info("Credentials loaded from %s.", path)
        return path

    raise FileNotFoundError(
        "No Google credentials found. Set GOOGLE_CREDENTIALS_JSON env var "
        f"or provide {path}."
    )


# ── Auth ──────────────────────────────────────────────────────────────────────
def _build_service():
    """
    Build the Drive service with:
      - cache_discovery=False  → skips the discovery doc network fetch (big speedup)
      - request_timeout        → fail fast instead of hanging forever
    """
    creds_path = _get_credentials_path()
    creds = service_account.Credentials.from_service_account_file(
        creds_path, scopes=SCOPES
    )

    # Pass credentials directly — works with google-auth >= 2.x
    # cache_discovery=False prevents the hanging discovery document GET request
    service = build(
        "drive", "v3",
        credentials=creds,
        cache_discovery=False,
    )
    return service


# ── Manifest (incremental sync) ───────────────────────────────────────────────
def _load_manifest() -> dict:
    if os.path.exists(MANIFEST_FILE):
        with open(MANIFEST_FILE, "r") as f:
            return json.load(f)
    return {}


def _save_manifest(manifest: dict) -> None:
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)


# ── Drive helpers ─────────────────────────────────────────────────────────────
def _list_files(service) -> list:
    """List all supported files visible to the service account."""
    mime_query = " or ".join(f"mimeType='{m}'" for m in SUPPORTED_MIME)
    results = service.files().list(
        pageSize=100,
        fields="files(id, name, mimeType, modifiedTime, md5Checksum)",
        q=f"({mime_query}) and trashed=false",
    ).execute()
    return results.get("files", [])


def _download_file(service, file_id: str, file_name: str, mime_type: str) -> str:
    """Download a single file; exports Google Docs as plain text. Returns saved filename."""
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    ext = SUPPORTED_MIME.get(mime_type, "")

    if mime_type == "application/vnd.google-apps.document":
        request = service.files().export_media(fileId=file_id, mimeType="text/plain")
        ext = ".txt"
    else:
        request = service.files().get_media(fileId=file_id)

    base_name = file_name if file_name.lower().endswith(ext) else file_name + ext
    file_path = os.path.join(DOWNLOAD_DIR, base_name)

    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()

    with open(file_path, "wb") as f:
        f.write(buf.getvalue())

    logger.info("Downloaded: %s", base_name)
    return base_name


# ── Main sync ─────────────────────────────────────────────────────────────────
def sync_drive_files() -> tuple[list, list]:
    """
    Sync files from Google Drive.
    Returns (downloaded_files, skipped_files) — lists of file names.
    """
    service  = _build_service()
    files    = _list_files(service)

    if not files:
        logger.info("No supported files found in Drive.")
        return [], []

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    manifest = _load_manifest()

    downloaded: list[str] = []
    skipped:    list[str] = []

    for file in files:
        file_id   = file["id"]
        file_name = file["name"]
        mime_type = file["mimeType"]
        modified  = file.get("modifiedTime", "")
        checksum  = file.get("md5Checksum") or modified   # Google Docs have no md5

        if file_id in manifest and manifest[file_id].get("checksum") == checksum:
            skipped.append(file_name)
            logger.info("Skipped (unchanged): %s", file_name)
            continue

        saved_name = _download_file(service, file_id, file_name, mime_type)
        downloaded.append(saved_name)
        manifest[file_id] = {
            "file_name": saved_name,
            "checksum":  checksum,
            "mime_type": mime_type,
            "modified":  modified,
        }

    _save_manifest(manifest)
    return downloaded, skipped