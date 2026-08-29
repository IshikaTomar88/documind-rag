"""
rag.py
"""

from __future__ import annotations

from llm import generate_answer
from vectorstore import VectorStore


def answer_question(store: VectorStore, question: str, top_k: int = 4,
                     llm_backend: str | None = None) -> dict:
    hits = store.query(question, top_k=top_k)
    answer = generate_answer(question, hits, backend=llm_backend)
    citations = [
        {
            "source": h["source"],
            "page": h["page"],
            "excerpt": (h["text"][:280] + "...") if len(h["text"]) > 280 else h["text"],
            "relevance": round(1 - h["distance"], 4) if h["distance"] is not None else None,
        }
        for h in hits
    ]
    return {"question": question, "answer": answer, "citations": citations}
