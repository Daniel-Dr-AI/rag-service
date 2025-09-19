# ingest.py — simple chunker, no LangChain needed
import os, glob, pickle
from typing import List

# Optional PDF support (works if you installed: py -m pip install pypdf)
try:
    from pypdf import PdfReader
    HAS_PDF = True
except Exception:
    HAS_PDF = False

DATA_DIR = "data"
OUT_FILE = "chunks.pkl"

def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_pdf_file(path: str) -> str:
    if not HAS_PDF:
        return ""
    try:
        reader = PdfReader(path)
        return "\n".join([(p.extract_text() or "") for p in reader.pages])
    except Exception:
        return ""

def simple_chunk(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    # character-based chunking; safe and simple
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + chunk_size)
        chunk = text[i:j].strip()
        if chunk:
            chunks.append(chunk)
        i = j - overlap if j - overlap > i else j
    return chunks

def main():
    if not os.path.isdir(DATA_DIR):
        print(f"No folder named '{DATA_DIR}' found.")
        return

    # collect all .txt and .pdf files in data/
    paths = []
    paths += glob.glob(os.path.join(DATA_DIR, "*.txt"))
    paths += glob.glob(os.path.join(DATA_DIR, "*.pdf"))

    if not paths:
        print(f"No .txt or .pdf files found in ./{DATA_DIR}. Add files and run again.")
        return

    all_chunks = []
    for p in paths:
        text = ""
        if p.lower().endswith(".txt"):
            text = read_text_file(p)
        elif p.lower().endswith(".pdf"):
            text = read_pdf_file(p)
        if not text.strip():
            continue
        all_chunks.extend(simple_chunk(text, chunk_size=500, overlap=100))

    if not all_chunks:
        print("No text content found in your files.")
        return

    with open(OUT_FILE, "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"Ingested {len(all_chunks)} chunks from {len(paths)} file(s). Saved to {OUT_FILE}.")

if __name__ == "__main__":
    main()
