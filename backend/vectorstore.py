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
requires reimplementing this one module -- nothing in rag.py or main.py
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
            {"source": c.source, "page": c.page or -1, "chunk_index": c.chunk_index}
            for c in chunks
        ]

        # TF-IDF must be (re)fit on the *full* accumulated corpus, not just
        # the new batch, or old vectors become incomparable to new ones.
        if isinstance(self.embedder, TfidfEmbedder):
            existing = self.collection.get(include=["documents"])
            corpus = (existing.get("documents") or []) + documents
            self.embedder.vectorizer.fit(corpus)
            self.embedder._fitted = True
            self._save_tfidf_state()
            # re-embed everything so old and new chunks share one vocabulary
            if existing.get("ids"):
                self.collection.delete(ids=existing["ids"])
                self.collection.add(
                    ids=existing["ids"], documents=existing["documents"],
                    metadatas=existing.get("metadatas") or [{} for _ in existing["ids"]],
                )

        self.collection.add(ids=ids, documents=documents, metadatas=metadatas)
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
        if TFIDF_STATE_PATH.exists():
            TFIDF_STATE_PATH.unlink()
