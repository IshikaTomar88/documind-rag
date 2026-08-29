"""
embeddings.py
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
        vectors = matrix.toarray()

        # BUG FIX: `max_features` is a CAP, not a fixed size -- if the
        # corpus has fewer unique (non-stopword) terms than max_features,
        # the vectorizer produces a narrower vector. Since the vocabulary
        # grows every time new documents are ingested and the vectorizer
        # is refit, two calls to embed() at different points in time can
        # return different widths. Chroma locks in the dimensionality of
        # the very first vector it stores, so any later batch at a
        # different width raises InvalidArgumentError and the whole
        # collection becomes unusable. We always pad/truncate to a fixed
        # `self.dimension` so every vector this embedder ever returns is
        # exactly the same width, regardless of vocabulary size.
        width = vectors.shape[1] if vectors.ndim == 2 else 0
        if width < self.dimension:
            pad = self.dimension - width
            vectors = [row.tolist() + [0.0] * pad for row in vectors]
        elif width > self.dimension:
            vectors = [row[: self.dimension].tolist() for row in vectors]
        else:
            vectors = vectors.tolist()

        return vectors


def get_embedder(backend: str | None = None) -> BaseEmbedder:
    backend = (backend or os.environ.get("EMBEDDING_BACKEND", "tfidf")).lower()
    if backend == "openai":
        return OpenAIEmbedder(model=os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    if backend == "local":
        return LocalEmbedder(model_name=os.environ.get("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    if backend == "tfidf":
        return TfidfEmbedder()
    raise ValueError(f"Unknown EMBEDDING_BACKEND: {backend}")
