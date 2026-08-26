"""
test_ingest.py
--------------
Run with: pytest backend/test_ingest.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ingest import chunk_text, clean_text


def test_clean_text_dehyphenates():
    raw = "This is infor-\nmation split across a line."
    assert "infor-\nmation" not in clean_text(raw)
    assert "information" in clean_text(raw)


def test_clean_text_collapses_newlines():
    raw = "Paragraph one.\n\n\n\nParagraph two."
    cleaned = clean_text(raw)
    assert "\n\n\n" not in cleaned


def test_chunk_text_respects_size():
    text = " ".join(f"word{i}" for i in range(1000))
    chunks = chunk_text(text, source="test.pdf", page=1, chunk_size_words=100, overlap_words=20)
    for c in chunks:
        assert len(c.text.split()) <= 100


def test_chunk_text_overlap():
    text = " ".join(f"word{i}" for i in range(300))
    chunks = chunk_text(text, source="test.pdf", page=1, chunk_size_words=100, overlap_words=20)
    # consecutive chunks should share the overlapping words
    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()
    overlap = set(first_words[-20:]) & set(second_words[:20])
    assert len(overlap) > 0


def test_chunk_text_empty_input():
    assert chunk_text("", source="test.pdf") == []


def test_chunk_metadata_is_attached():
    text = "hello world " * 50
    chunks = chunk_text(text, source="handbook.pdf", page=3, chunk_size_words=20, overlap_words=5)
    assert all(c.source == "handbook.pdf" for c in chunks)
    assert all(c.page == 3 for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
