from fastapi import FastAPI, Form
from fastapi.responses import Response
from pydantic import BaseModel
from twilio.twiml.messaging_response import MessagingResponse
import agent
import os
import logging

# ------------------------
# FastAPI app
# ------------------------
app = FastAPI(title="Bilingual RAG Agent")

# ------------------------
# Logging setup
# ------------------------
logging.basicConfig(level=logging.INFO)

# Check for API key (safe check)
if os.getenv("OPENAI_API_KEY"):
    logging.info("OPENAI_API_KEY loaded ✅")
else:
    logging.warning("OPENAI_API_KEY is NOT set ❌ — embeddings will fail.")

# ------------------------
# Schema for /ask
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
    return result  # return full structured JSON for API clients

# ------------------------
# Twilio SMS webhook
# ------------------------
@app.post("/sms")
async def sms_reply(From: str = Form(...), Body: str = Form(...)):
    # Run the agent with explicit instruction
    result = agent.agent_run(
        user_query=Body + " (Answer in one short sentence.)",
        max_steps=6,
        top_k=5,
        model="gpt-4o-mini"
    )

    final_answer = result.get("final", {}).get("answer", {}).get(
        "english",
        "Sorry, I couldn’t process that."
    )

    # 🪓 Force single sentence by cutting at first period
    if "." in final_answer:
        final_answer = final_answer.split(".")[0].strip() + "."

    # Safety: trim to 200 chars max
    final_answer = final_answer[:200]

    # Build Twilio-compatible XML response
    resp = MessagingResponse()
    resp.message(final_answer)

    return Response(content=str(resp), media_type="application/xml")


# ------------------------
# Optional root route
# ------------------------
@app.get("/")
def root():
    return {"message": "Bilingual RAG Agent is live!"}
