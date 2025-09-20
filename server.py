from fastapi import FastAPI, Form
from fastapi.responses import Response
from pydantic import BaseModel
from twilio.twiml.messaging_response import MessagingResponse
import agent
import json
import logging

# ------------------------
# FastAPI app
# ------------------------
app = FastAPI(title="Bilingual RAG Agent")

# ------------------------
# Logging setup
# ------------------------
logging.basicConfig(level=logging.INFO)

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
    try:
        result = agent.agent_run(
            user_query=request.question,
            max_steps=request.steps,
            top_k=request.top_k,
            model=request.model
        )
        return result  # full JSON for API clients
    except Exception as e:
        logging.exception("Error in /ask endpoint")
        return {"status": "error", "message": str(e)}

# ------------------------
# Twilio SMS webhook
# ------------------------
@app.post("/sms")
async def sms_reply(From: str = Form(...), Body: str = Form(...)):
    try:
        # Run the agent pipeline
        result = agent.agent_run(
            user_query=Body,
            max_steps=6,
            top_k=5,
            model="gpt-4o-mini"
        )

        # Log the full result for debugging (won’t be sent to user)
        logging.info("Agent result: %s", json.dumps(result, indent=2))

        # Extract just the English answer
        final_answer = (
            result.get("final", {})
                  .get("answer", {})
                  .get("english", "Sorry, I couldn’t process that.")
        )

    except Exception as e:
        logging.exception("Error in /sms endpoint")
        final_answer = "Something went wrong, please try again later."

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
