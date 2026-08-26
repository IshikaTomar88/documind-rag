"""
embeddings.py
-------------
Pluggable embedding backends behind one common interface.

Why pluggable rather than hard-coded to one provider:
  - "openai"     -> best quality, needs an API key, costs money per call.
  - "local"      -> sentence-transformers running on the client's own
                     machine. No API key, no per-call cost, data never
                     leaves their infrastructure (important for clients
                     with confidential internal documents).
  - "tfidf"      -> zero-dependency, zero-cost, fully offline fallback.
                     Not semantic (it's a classic bag-of-words vectorizer),
                     but it lets the whole pipeline run and be demoed with
                     no API key and no multi-GB model download -- useful
                     for local dev, CI, and a client's first trial run.

The embedding backend is chosen once via EMBEDDING_BACKEND and every
piece downstream (vectorstore, ingestion CLI) just calls `.embed(texts)`
without caring which one is active.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    dimension: int

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, model: str = "text-embedding-3-small"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model
        self.dimension = 1536

    def embed(self, texts: list[str]) -> list[list[float]]:
        # OpenAI's embeddings endpoint accepts batches directly.
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]


class LocalEmbedder(BaseEmbedder):
    """sentence-transformers, runs fully on the local machine/GPU/CPU."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return vectors.tolist()


class TfidfEmbedder(BaseEmbedder):
    """
    Zero-dependency-beyond-scikit-learn fallback. Fit lazily on the first
    batch of documents seen (typically the corpus being ingested), then
    reused for query-time embedding via `.transform()`.

    Not semantic search in the deep-learning sense, but it is a real,
    working retrieval signal (term frequency / inverse document
    frequency + cosine similarity) with no external calls and no model
    download -- good enough to prove the whole pipeline end-to-end
    offline, and a fine default for small, vocabulary-narrow document
    sets (e.g. a single product manual).
    """

    def __init__(self, max_features: int = 4096):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
        self.dimension = max_features
        self._fitted = False

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self._fitted:
            matrix = self.vectorizer.fit_transform(texts)
            self._fitted = True
        else:
            matrix = self.vectorizer.transform(texts)
        # scikit-learn's TfidfVectorizer already L2-normalizes rows by
        # default (norm="l2"), which is what makes cosine distance in the
        # vector store meaningful -- an unnormalized vector would make
        # "distance" scale with document length rather than topical
        # similarity.
        return matrix.toarray().tolist()


def get_embedder(backend: str | None = None) -> BaseEmbedder:
    backend = (backend or os.environ.get("EMBEDDING_BACKEND", "tfidf")).lower()
    if backend == "openai":
        return OpenAIEmbedder(model=os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    if backend == "local":
        return LocalEmbedder(model_name=os.environ.get("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    if backend == "tfidf":
        return TfidfEmbedder()
    raise ValueError(f"Unknown EMBEDDING_BACKEND: {backend}")
