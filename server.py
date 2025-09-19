from fastapi import FastAPI, Form
from fastapi.responses import Response
from pydantic import BaseModel
from twilio.twiml.messaging_response import MessagingResponse
import agent

app = FastAPI(title="Bilingual RAG Agent")

# ------------------------
# API schema for /ask
# ------------------------
class QueryRequest(BaseModel):
    question: str
    steps: int = 6
    top_k: int = 5
    model: str = "gpt-4o-mini"

# ------------------------
# Health check
# ------------------------
@app.get("/health")
def health():
    return {"status": "running"}

# ------------------------
# Main chatbot API
# ------------------------
@app.post("/ask")
def ask(request: QueryRequest):
    result = agent.agent_run(
        user_query=request.question,
        max_steps=request.steps,
        top_k=request.top_k,
        model=request.model
    )
    return {"answer": result}

# ------------------------
# Twilio SMS webhook
# ------------------------
@app.post("/sms")
async def sms_reply(From: str = Form(...), Body: str = Form(...)):
    # Pass the incoming SMS text into your RAG agent
    result = agent.agent_run(
        user_query=Body,
        max_steps=6,
        top_k=5,
        model="gpt-4o-mini"
    )

    # Build a Twilio-compatible XML response
    resp = MessagingResponse()
    resp.message(result)

    return Response(content=str(resp), media_type="application/xml")

# ------------------------
# Optional root route
# ------------------------
@app.get("/")
def root():
    return {"message": "Bilingual RAG Agent is live!"}
