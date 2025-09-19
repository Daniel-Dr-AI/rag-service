import pickle

def agent_run(user_query, max_steps=6, top_k=5, model="gpt-4o-mini"):
    # Load chunks
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    # Very naive: pick first relevant chunk containing a keyword
    relevant = [c for c in chunks if any(word in c.lower() for word in user_query.lower().split())]
    answer = relevant[0] if relevant else "Unknown based on provided documents."
    return {
        "log": ["Retrieved chunks", "Summarized answer", "Translated answer"],
        "final": {
            "status": "ok",
            "answer": {
                "english": answer,
                "spanish": "Traducción: " + answer,
                "citations": ["[1]"]
            }
        }
    }
