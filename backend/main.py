"""
main.py
-------
FastAPI backend for DocuMind.

Run:
    uvicorn main:app --reload --port 8000

Endpoints
---------
POST /documents/upload   -> ingest one or more PDFs/text files
GET  /documents           -> list ingested source documents
POST /ask                  -> ask a question, get an answer + citations
DELETE /documents          -> wipe the index and start over
GET  /health                -> liveness check

The frontend (Streamlit) talks to this service over plain HTTP, which
means the backend can also be called from a client's own internal tools,
a Slack bot, a mobile app, etc. -- it isn't locked to one UI.
"""

from __future__ import annotations

import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ingest import ingest_any
from rag import answer_question
from vectorstore import VectorStore

app = FastAPI(
    title="DocuMind API",
    description="Vertical RAG document question-answering backend.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your frontend's origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)

_store: VectorStore | None = None


def get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore(backend=os.environ.get("EMBEDDING_BACKEND"))
    return _store


class AskRequest(BaseModel):
    question: str
    top_k: int = 4


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[dict]


class UploadResponse(BaseModel):
    filename: str
    chunks_added: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/documents/upload", response_model=list[UploadResponse])
async def upload_documents(files: list[UploadFile] = File(...)):
    store = get_store()
    results = []
    for file in files:
        content = await file.read()
        try:
            chunks = ingest_any(content, file.filename)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        n = store.add_chunks(chunks)
        results.append(UploadResponse(filename=file.filename, chunks_added=n))
    return results


@app.get("/documents")
def list_documents():
    store = get_store()
    return {"sources": store.list_sources(), "total_chunks": store.stats()["total_chunks"]}


@app.delete("/documents")
def clear_documents():
    store = get_store()
    store.reset()
    return {"status": "cleared"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    store = get_store()
    if store.stats()["total_chunks"] == 0:
        raise HTTPException(status_code=400, detail="No documents have been uploaded yet.")
    result = answer_question(store, req.question, top_k=req.top_k)
    return result
