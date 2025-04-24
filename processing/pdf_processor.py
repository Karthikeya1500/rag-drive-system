from pypdf import PdfReader


def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        text += page.extract_text() + "\n"

    return text


def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        start += (chunk_size - overlap)

    return chunks


def process_pdfs(folder_path="downloads"):
    import os

    all_chunks = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(folder_path, file_name)

            text = extract_text_from_pdf(file_path)

            chunks = chunk_text(text)

            for chunk in chunks:
                all_chunks.append({
                    "text": chunk,
                    "metadata": {
                        "file_name": file_name
                    }
                })

    return all_chunks