"""
vectorstore.py
--------------
Thin wrapper around ChromaDB that:
  - persists to disk (so the index survives an app restart)
  - stores rich metadata per chunk (source filename, page number, chunk
    index) so retrieval results can be cited precisely
  - handles TF-IDF vectorizer persistence as a special case, since that
    embedder (unlike OpenAI/local transformer models) has state that must
    be fit once and reused consistently between ingestion and query time

Swapping the vector store itself (e.g. to FAISS, Pinecone, Weaviate) only
requires reimplementing this one module -- nothing in rag.py or app.py
needs to change, because they only depend on this file's public functions.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import chromadb
from chromadb.api.types import EmbeddingFunction

from embeddings import BaseEmbedder, TfidfEmbedder, get_embedder
from ingest import Chunk

PERSIST_DIR = Path(__file__).parent / "chroma_store"
TFIDF_STATE_PATH = PERSIST_DIR / "tfidf_vectorizer.pkl"
COLLECTION_NAME = "documind_chunks"


class _ChromaEmbeddingAdapter(EmbeddingFunction):
    """Adapts our BaseEmbedder interface to Chroma's expected callable."""

    def __init__(self, embedder: BaseEmbedder):
        self.embedder = embedder

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self.embedder.embed(list(input))


class VectorStore:
    def __init__(self, backend: str | None = None):
        PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        self.embedder = get_embedder(backend)

        # TF-IDF's vectorizer has to be fit consistently across ingest and
        # query calls, and (unlike the neural embedders) that fitted state
        # doesn't exist anywhere except in memory -- so we persist it
        # ourselves alongside the Chroma index.
        if isinstance(self.embedder, TfidfEmbedder) and TFIDF_STATE_PATH.exists():
            with open(TFIDF_STATE_PATH, "rb") as f:
                self.embedder.vectorizer = pickle.load(f)
                self.embedder._fitted = True

        self.client = chromadb.PersistentClient(path=str(PERSIST_DIR))
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=_ChromaEmbeddingAdapter(self.embedder),
            # Cosine distance makes "relevance" scores (1 - distance)
            # meaningful and comparable across embedding backends, unlike
            # raw L2 distance which is sensitive to vector magnitude.
            metadata={"hnsw:space": "cosine"},
        )

    def _save_tfidf_state(self) -> None:
        if isinstance(self.embedder, TfidfEmbedder):
            with open(TFIDF_STATE_PATH, "wb") as f:
                pickle.dump(self.embedder.vectorizer, f)

    def add_chunks(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0

        ids = [f"{c.source}::p{c.page}::{c.chunk_index}" for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "source": c.source,
                "page": c.page if c.page is not None else -1,
                "chunk_index": c.chunk_index,
            }
            for c in chunks
        ]

        if isinstance(self.embedder, TfidfEmbedder):
            # TF-IDF must be (re)fit on the *full* accumulated corpus, not
            # just the new batch, or old vectors become incomparable to
            # new ones (different vocabulary => different vector meaning).
            #
            # BUG FIX: the original call was
            #   self.collection.get(include=["documents"])
            # which does NOT return metadata (Chroma only returns what you
            # ask for in `include`). Every re-fit was silently wiping the
            # source/page/chunk_index of every previously-indexed chunk,
            # breaking citations for anything uploaded before the most
            # recent batch. We now explicitly request both.
            existing = self.collection.get(include=["documents", "metadatas"])
            existing_ids = existing.get("ids") or []
            existing_docs = existing.get("documents") or []
            existing_metas = existing.get("metadatas") or []

            corpus = existing_docs + documents
            self.embedder.vectorizer.fit(corpus)
            self.embedder._fitted = True
            self._save_tfidf_state()

            # Re-embed every existing chunk under the newly fitted
            # vocabulary so old and new vectors stay comparable.
            if existing_ids:
                self.collection.upsert(ids=existing_ids, documents=existing_docs, metadatas=existing_metas)

        # BUG FIX: this was `self.collection.add(...)`. Chroma's `add()`
        # raises if any id already exists, so re-uploading a file you'd
        # already indexed (same filename => same ids) crashed the app.
        # `upsert()` is idempotent: same id updates in place, new id inserts.
        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        self._save_tfidf_state()
        return len(chunks)

    def query(self, question: str, top_k: int = 4) -> list[dict]:
        results = self.collection.query(query_texts=[question], n_results=top_k)
        hits = []
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            hits.append({
                "text": doc,
                "source": meta.get("source"),
                "page": meta.get("page"),
                "chunk_index": meta.get("chunk_index"),
                "distance": dist,
            })
        return hits

    def stats(self) -> dict:
        return {"total_chunks": self.collection.count()}

    def list_sources(self) -> list[str]:
        data = self.collection.get(include=["metadatas"])
        sources = {m.get("source") for m in (data.get("metadatas") or []) if m.get("source")}
        return sorted(sources)

    def reset(self) -> None:
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=_ChromaEmbeddingAdapter(self.embedder),
            metadata={"hnsw:space": "cosine"},
        )
        if isinstance(self.embedder, TfidfEmbedder):
            self.embedder.vectorizer = type(self.embedder.vectorizer)(
                max_features=self.embedder.dimension, stop_words="english"
            )
            self.embedder._fitted = False
            if TFIDF_STATE_PATH.exists():
                TFIDF_STATE_PATH.unlink()
