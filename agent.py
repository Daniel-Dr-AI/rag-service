import faiss
import pickle
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ------------------------
# Config
# ------------------------
INDEX_FILE = Path("vectorstore/faiss.index")
CHUNKS_FILE = Path("vectorstore/chunks.pkl")

# Embedding model for queries
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBEDDING_MODEL)

# Load FAISS index + chunks into memory
if not INDEX_FILE.exists() or not CHUNKS_FILE.exists():
    raise FileNotFoundError("Vectorstore not built yet. Run ingest.py first!")

index = faiss.read_index(str(INDEX_FILE))
with open(CHUNKS_FILE, "rb") as f:
    chunks = pickle.load(f)

# OpenAI client for generation (still needed for final answers)
client = OpenAI()

# ------------------------
# Search function
# ------------------------
def search(query: str, top_k: int = 5):
    q_vec = embedder.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(q_vec, top_k)
    results = [chunks[i] for i in indices[0] if i < len(chunks)]
    return results

# ------------------------
# Agent run function
# ------------------------
def agent_run(user_query, max_steps=6, top_k=5, model="gpt-4o-mini"):
    # Retrieve top context chunks
    context_chunks = search(user_query, top_k=top_k)
    context = "\n".join(context_chunks)

    # Construct prompt
    prompt = f"""You are a helpful assistant.
Use the following context to answer the question.

Context:
{context}

Question: {user_query}
Answer:"""

    # Generate answer with OpenAI chat model
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    final_answer = completion.choices[0].message.content.strip()

    return {
        "final": {"answer": {"english": final_answer}},
        "context_used": context_chunks
    }
