"""
document_processor.py
Handles extraction and chunking for PDF, TXT, and DOCX files.
Attaches full metadata: doc_id, file_name, source, page, chunk_index, file_type.
"""

import os
import re
import hashlib

from pypdf import PdfReader

try:
    import docx as python_docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False


# ── Text extraction ───────────────────────────────────────────────────────────

def extract_text_from_pdf(file_path: str) -> list[dict]:
    """Returns list of {text, page} dicts, one per PDF page."""
    reader = PdfReader(file_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"text": text.strip(), "page": i + 1})
    return pages


def extract_text_from_txt(file_path: str) -> list[dict]:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return [{"text": text.strip(), "page": None}]


def extract_text_from_docx(file_path: str) -> list[dict]:
    if not DOCX_AVAILABLE:
        print("python-docx not installed; skipping DOCX file.")
        return []
    doc = python_docx.Document(file_path)
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return [{"text": text, "page": None}]


# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Normalise whitespace and remove non-printable characters."""
    text = re.sub(r"[ \t]+", " ", text)            # collapse horizontal whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)          # collapse 3+ blank lines → 2
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)    # strip non-ASCII (keep newlines)
    return text.strip()


# ── Paragraph-aware chunking with overlap ─────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 600, overlap: int = 120) -> list[str]:
    """
    Paragraph-aware chunker:
      1. Splits on paragraph boundaries (double newlines).
      2. If a paragraph exceeds chunk_size, splits further by sentence.
      3. Applies character-level overlap between consecutive chunks.
    """
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    raw_chunks: list[str] = []
    current = ""

    for para in paragraphs:
        # Case: para fits in current buffer
        if len(current) + len(para) + 2 <= chunk_size:
            current = (current + "\n\n" + para).strip() if current else para
            continue

        # Flush current buffer
        if current:
            raw_chunks.append(current)
            current = ""

        # Case: paragraph itself is too long → split by sentence
        if len(para) > chunk_size:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            buf = ""
            for sent in sentences:
                if len(buf) + len(sent) + 1 <= chunk_size:
                    buf = (buf + " " + sent).strip() if buf else sent
                else:
                    if buf:
                        raw_chunks.append(buf)
                    buf = sent
            current = buf
        else:
            current = para

    if current:
        raw_chunks.append(current)

    # Apply overlap: prepend tail of previous chunk to each chunk
    if overlap <= 0 or len(raw_chunks) < 2:
        return raw_chunks

    overlapped: list[str] = [raw_chunks[0]]
    for i in range(1, len(raw_chunks)):
        tail = raw_chunks[i - 1][-overlap:]
        overlapped.append((tail + " " + raw_chunks[i]).strip())

    return overlapped


# ── Stable doc ID ─────────────────────────────────────────────────────────────

def make_doc_id(file_path: str) -> str:
    """Deterministic ID derived from the file path (MD5 hex)."""
    return hashlib.md5(file_path.encode()).hexdigest()


# ── Main pipeline ─────────────────────────────────────────────────────────────

def process_documents(folder_path: str = "downloads") -> list[dict]:
    """
    Process all supported documents in folder_path.

    Returns a list of chunk dicts:
        {
            "text": str,
            "metadata": {
                "doc_id":      str,
                "file_name":   str,
                "source":      "gdrive",
                "page":        int | None,
                "chunk_index": int,
                "file_type":   str,
            }
        }
    """
    all_chunks: list[dict] = []

    for file_name in sorted(os.listdir(folder_path)):
        # Skip hidden / manifest files
        if file_name.startswith("."):
            continue

        file_path = os.path.join(folder_path, file_name)
        ext = file_name.lower().rsplit(".", 1)[-1]

        if ext == "pdf":
            pages = extract_text_from_pdf(file_path)
        elif ext == "txt":
            pages = extract_text_from_txt(file_path)
        elif ext == "docx":
            pages = extract_text_from_docx(file_path)
        else:
            continue  # unsupported type

        doc_id = make_doc_id(file_path)

        for page_info in pages:
            raw = page_info["text"]
            page_num = page_info["page"]

            if not raw.strip():
                continue

            cleaned = clean_text(raw)
            chunks = chunk_text(cleaned)

            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "text": chunk,
                    "metadata": {
                        "doc_id":      doc_id,
                        "file_name":   file_name,
                        "source":      "gdrive",
                        "page":        page_num,
                        "chunk_index": i,
                        "file_type":   ext,
                    },
                })

    print(f"Processed {len(all_chunks)} chunks from {folder_path}")
    return all_chunks
