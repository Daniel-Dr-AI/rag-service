from fastapi import FastAPI
from pydantic import BaseModel
import agent

app = FastAPI(title="Bilingual RAG Agent")

class QueryRequest(BaseModel):
    question: str
    steps: int = 6
    top_k: int = 5
    model: str = "gpt-4o-mini"

@app.get("/health")
def health():
    return {"status": "running"}

@app.post("/ask")
def ask(request: QueryRequest):
    result = agent.agent_run(
        user_query=request.question,
        max_steps=request.steps,
        top_k=request.top_k,
        model=request.model
    )
    return result
