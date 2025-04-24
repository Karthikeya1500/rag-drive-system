import io
import os
import json

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ── Config ────────────────────────────────────────────────────────────────────
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
SERVICE_ACCOUNT_FILE = "credentials.json"
DOWNLOAD_DIR = "downloads"
MANIFEST_FILE = os.path.join(DOWNLOAD_DIR, ".manifest.json")

# MIME types the system supports and their local extensions
SUPPORTED_MIME = {
    "application/pdf": ".pdf",
    "application/vnd.google-apps.document": ".txt",   # exported as plain text
    "text/plain": ".txt",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
}


# ── Auth ──────────────────────────────────────────────────────────────────────
def authenticate_drive():
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    return build("drive", "v3", credentials=creds)


# ── Manifest (incremental sync) ───────────────────────────────────────────────
def load_manifest() -> dict:
    if os.path.exists(MANIFEST_FILE):
        with open(MANIFEST_FILE, "r") as f:
            return json.load(f)
    return {}


def save_manifest(manifest: dict):
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)


# ── Drive helpers ─────────────────────────────────────────────────────────────
def list_files(service) -> list:
    """List all supported files visible to the service account."""
    mime_query = " or ".join(f"mimeType='{m}'" for m in SUPPORTED_MIME)
    results = service.files().list(
        pageSize=100,
        fields="files(id, name, mimeType, modifiedTime, md5Checksum)",
        q=f"({mime_query}) and trashed=false",
    ).execute()
    return results.get("files", [])


def download_file(service, file_id: str, file_name: str, mime_type: str) -> str:
    """Download a single file; exports Google Docs as plain text. Returns saved filename."""
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    ext = SUPPORTED_MIME.get(mime_type, "")

    if mime_type == "application/vnd.google-apps.document":
        # Export Google Doc → plain text
        request = service.files().export_media(fileId=file_id, mimeType="text/plain")
        ext = ".txt"
    else:
        request = service.files().get_media(fileId=file_id)

    # Ensure file name has the correct extension
    base_name = file_name
    if not base_name.lower().endswith(ext):
        base_name = base_name + ext

    file_path = os.path.join(DOWNLOAD_DIR, base_name)

    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()

    with open(file_path, "wb") as f:
        f.write(buf.getvalue())

    print(f"Downloaded: {base_name}")
    return base_name


# ── Main sync ─────────────────────────────────────────────────────────────────
def sync_drive_files() -> tuple[list, list]:
    """
    Sync files from Google Drive.

    Returns:
        (downloaded_files, skipped_files) — lists of file names.
    """
    service = authenticate_drive()
    files = list_files(service)

    if not files:
        print("No supported files found.")
        return [], []

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    manifest = load_manifest()

    downloaded: list[str] = []
    skipped: list[str] = []

    for file in files:
        file_id    = file["id"]
        file_name  = file["name"]
        mime_type  = file["mimeType"]
        modified   = file.get("modifiedTime", "")
        # Google Docs don't have md5; fall back to modifiedTime as change signal
        checksum   = file.get("md5Checksum") or modified

        # ── Incremental sync: skip unchanged files ────────────────────────────
        if file_id in manifest and manifest[file_id].get("checksum") == checksum:
            skipped.append(file_name)
            print(f"Skipped (unchanged): {file_name}")
            continue

        saved_name = download_file(service, file_id, file_name, mime_type)
        downloaded.append(saved_name)

        manifest[file_id] = {
            "file_name":  saved_name,
            "checksum":   checksum,
            "mime_type":  mime_type,
            "modified":   modified,
        }

    save_manifest(manifest)
    return downloaded, skipped