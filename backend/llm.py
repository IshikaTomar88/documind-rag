"""
llm.py
------
Pluggable LLM backend for the "Generation" half of RAG.

Supports Anthropic (Claude) and OpenAI via API key, plus an
"extractive" offline fallback that requires no API key at all --
it just returns the most relevant retrieved sentences directly.
That fallback exists so:
  1. The full pipeline can be demoed/tested with zero API cost.
  2. If a client's API key is missing/invalid/rate-limited, the app
     degrades gracefully instead of crashing.
"""

from __future__ import annotations

import os
import re

SYSTEM_PROMPT = (
    "You are a precise document question-answering assistant. "
    "Answer the user's question using ONLY the provided context excerpts. "
    "If the answer is not contained in the context, say clearly that the "
    "documents don't contain that information -- never guess or use "
    "outside knowledge. Keep answers concise and factual. When you state "
    "a fact, prefer language that makes it easy for the reader to see "
    "which excerpt it came from."
)


def _build_context_block(hits: list[dict]) -> str:
    parts = []
    for i, hit in enumerate(hits, start=1):
        page_info = f", page {hit['page']}" if hit.get("page") not in (None, -1) else ""
        parts.append(f"[Excerpt {i} — {hit['source']}{page_info}]\n{hit['text']}")
    return "\n\n".join(parts)


def generate_answer(question: str, hits: list[dict], backend: str | None = None) -> str:
    backend = (backend or os.environ.get("LLM_BACKEND", "extractive")).lower()

    if not hits:
        return "I couldn't find anything relevant to that question in the uploaded documents."

    if backend == "anthropic":
        return _generate_anthropic(question, hits)
    if backend == "openai":
        return _generate_openai(question, hits)
    return _generate_extractive(question, hits)


def _generate_anthropic(question: str, hits: list[dict]) -> str:
    import anthropic
    client = anthropic.Anthropic()
    context = _build_context_block(hits)
    message = client.messages.create(
        model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}",
        }],
    )
    return "".join(block.text for block in message.content if block.type == "text")


def _generate_openai(question: str, hits: list[dict]) -> str:
    from openai import OpenAI
    client = OpenAI()
    context = _build_context_block(hits)
    resp = client.chat.completions.create(
        model=os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        max_tokens=600,
    )
    return resp.choices[0].message.content


def _generate_extractive(question: str, hits: list[dict]) -> str:
    """
    No-API-key fallback: pick the sentences within the top retrieved
    chunks that share the most keywords with the question, and return
    them directly with their source tagged. This is intentionally
    conservative -- it never invents anything not present in the text.
    """
    question_words = set(re.findall(r"\w+", question.lower())) - _STOPWORDS
    scored_sentences: list[tuple[float, str, dict]] = []

    for hit in hits:
        sentences = re.split(r"(?<=[.!?])\s+", hit["text"])
        for sent in sentences:
            words = set(re.findall(r"\w+", sent.lower()))
            overlap = len(words & question_words)
            if overlap:
                scored_sentences.append((overlap, sent.strip(), hit))

    if not scored_sentences:
        top = hits[0]
        page_info = f" (p.{top['page']})" if top.get("page") not in (None, -1) else ""
        return (
            "I couldn't find an exact match, but the most relevant excerpt is from "
            f"{top['source']}{page_info}:\n\n\"{top['text'][:400]}...\""
        )

    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    best = scored_sentences[:3]

    lines = []
    for _, sent, hit in best:
        page_info = f", p.{hit['page']}" if hit.get("page") not in (None, -1) else ""
        lines.append(f"- {sent} (source: {hit['source']}{page_info})")

    return "Based on the documents:\n" + "\n".join(lines)


_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "what", "who", "when",
    "where", "why", "how", "does", "do", "did", "in", "on", "of", "to",
    "for", "and", "or", "this", "that", "with", "it", "as", "be", "by",
}
