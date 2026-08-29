"""
ingest.py
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass

from pypdf import PdfReader


@dataclass
class Chunk:
    text: str
    source: str          # original filename
    page: int | None      # page number, 1-indexed (None if not applicable)
    chunk_index: int      # position of this chunk within the document


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------

def extract_pdf_pages(file_bytes: bytes) -> list[tuple[int, str]]:
    """Returns a list of (page_number, raw_text) tuples, 1-indexed."""
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append((i, text))
    return pages


def clean_text(text: str) -> str:
    """
    Collapse excess whitespace, remove common PDF extraction artifacts
    (stray hyphenation at line breaks, repeated newlines) without
    destroying paragraph structure.
    """
    # de-hyphenate words split across a line break: "informa-\ntion" -> "information"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------

def chunk_text(
    text: str,
    source: str,
    page: int | None = None,
    chunk_size_words: int = 250,
    overlap_words: int = 40,
    start_index: int = 0,
) -> list[Chunk]:
    """
    Split `text` into overlapping word-count windows.

    chunk_size_words: target chunk length. 250 words (~350-400 tokens for
        English text) is a good default balance -- large enough to hold a
        full idea/paragraph for context, small enough that the LLM's
        context window can hold several retrieved chunks at once.
    overlap_words: how many words are repeated at the start of the next
        chunk, so a fact sitting on a chunk boundary is fully present in
        at least one chunk.
    """
    words = text.split()
    if not words:
        return []

    chunks: list[Chunk] = []
    step = max(chunk_size_words - overlap_words, 1)
    idx = start_index

    for start in range(0, len(words), step):
        window = words[start:start + chunk_size_words]
        if not window:
            break
        chunk_str = " ".join(window)
        chunks.append(Chunk(text=chunk_str, source=source, page=page, chunk_index=idx))
        idx += 1
        if start + chunk_size_words >= len(words):
            break

    return chunks


def ingest_pdf(
    file_bytes: bytes,
    filename: str,
    chunk_size_words: int = 250,
    overlap_words: int = 40,
) -> list[Chunk]:
    """Full pipeline: PDF bytes -> cleaned, chunked, page-tagged text."""
    pages = extract_pdf_pages(file_bytes)
    all_chunks: list[Chunk] = []
    running_index = 0

    for page_num, raw_text in pages:
        text = clean_text(raw_text)
        if not text:
            continue
        page_chunks = chunk_text(
            text, source=filename, page=page_num,
            chunk_size_words=chunk_size_words, overlap_words=overlap_words,
            start_index=running_index,
        )
        all_chunks.extend(page_chunks)
        running_index += len(page_chunks)

    return all_chunks


def ingest_text(
    file_bytes: bytes,
    filename: str,
    chunk_size_words: int = 250,
    overlap_words: int = 40,
) -> list[Chunk]:
    """Same pipeline for plain .txt/.md files (no page numbers)."""
    text = clean_text(file_bytes.decode("utf-8", errors="ignore"))
    return chunk_text(text, source=filename, page=None,
                       chunk_size_words=chunk_size_words, overlap_words=overlap_words)


def ingest_any(file_bytes: bytes, filename: str, **kwargs) -> list[Chunk]:
    name = filename.lower()
    if name.endswith(".pdf"):
        return ingest_pdf(file_bytes, filename, **kwargs)
    if name.endswith((".txt", ".md")):
        return ingest_text(file_bytes, filename, **kwargs)
    raise ValueError(f"Unsupported document type: {filename}")
