# RAG Service Deployment Guide

## Setup
1. Create virtual env: `python -m venv .venv`
2. Activate it:
   - Mac/Linux: `source .venv/bin/activate`
   - Windows: `.venv\Scripts\Activate`
3. Install deps: `pip install -r requirements.txt`
4. Copy `.env.example` to `.env` and paste your OpenAI API key.

## Build index
`python ingest.py`

## Run server
`uvicorn server:app --reload --port 8000`

## Test
```
curl http://127.0.0.1:8000/health
curl -X POST "http://127.0.0.1:8000/ask" -H "Content-Type: application/json" -d '{"question":"How many hours of sleep do adults need?"}'
```
