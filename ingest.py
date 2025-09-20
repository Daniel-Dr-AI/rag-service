import os
from pathlib import Path
import pickle
import numpy as np

from pypdf import PdfReader
import docx
from sentence_transformers import SentenceTransformer
import faiss

# ------------------------
# Config
# ------------------------
DATA_FOLDER = Path("data")
INDEX_FOLDER = Path("vectorstore")
INDEX_FILE = INDEX_FOLDER / "faiss.index"
CHUNKS_FILE = INDEX_FOLDER / "chunks.pkl"

# Use Hugging Face embeddings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBEDDING_MODEL)

# ------------------------
# Load and extract text
# ------------------------
def load_documents(folder: Path):
    docs = []
    for path in folder.glob("*"):
        if path.suffix.lower() == ".pdf":
            print(f"Loading PDF: {path.name}")
            reader = PdfReader(str(path))
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    docs.append(text)
        elif path.suffix.lower() == ".docx":
            print(f"Loading DOCX: {path.name}")
            doc = docx.Document(str(path))
            full_text = "\n".join([p.text for p in doc.paragraphs])
            if full_text.strip():
                docs.append(full_text)
    return docs

# ------------------------
# Chunking
# ------------------------
def chunk_texts(texts, chunk_size=1000, overlap=100):
    chunks = []
    for text in texts:
        text = text.replace("\n", " ")
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            chunks.append(chunk)
    return chunks

# ------------------------
# Embeddings
# ------------------------
def embed_chunks(chunks):
    embeddings = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    return embeddings.astype("float32")

# ------------------------
# Build or update FAISS
# ------------------------
def build_or_update_index(vectors, chunks):
    dim = vectors.shape[1]
    INDEX_FOLDER.mkdir(exist_ok=True)

    if INDEX_FILE.exists() and CHUNKS_FILE.exists():
        # load existing index and chunks
        index = faiss.read_index(str(INDEX_FILE))
        with open(CHUNKS_FILE, "rb") as f:
            old_chunks = pickle.load(f)

        # append new vectors + chunks
        index.add(vectors)
        all_chunks = old_chunks + chunks
    else:
        # fresh index
        index = faiss.IndexFlatL2(dim)
        index.add(vectors)
        all_chunks = chunks

    # save index + chunks
    faiss.write_index(index, str(INDEX_FILE))
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"Index now contains {index.ntotal} vectors.")

# ------------------------
# Main
# ------------------------
if __name__ == "__main__":
    texts = load_documents(DATA_FOLDER)
    print(f"Loaded {len(texts)} documents.")

    chunks = chunk_texts(texts)
    print(f"Split into {len(chunks)} chunks.")

    vectors = embed_chunks(chunks)
    print(f"Created {vectors.shape[0]} embeddings.")

    build_or_update_index(vectors, chunks)
    print("Ingestion complete.")
