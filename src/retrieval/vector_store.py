"""Persistent vector database (Chroma) behind an abstract interface.

What this file does (Member 2 deliverable):
  * turns Member 1's chunks into embeddings,
  * stores embeddings + text + metadata (source, category, page) in Chroma on disk,
  * runs semantic similarity search and returns the matched vectors as well,
    so the retriever can show the dot products and magnitudes behind each score.

Why not ``chromadb.Client()`` + ``query_texts=`` like the lab notebook?
  * ``Client()`` lives in RAM - everything is lost when the runtime restarts.
    ``PersistentClient`` writes to disk (coursework requirement 3: vector persistence).
  * ``query_texts`` embeds the question inside Chroma, so we never see the vectors.
    We embed ourselves and pass ``query_embeddings`` to keep the maths visible.
  * Chroma's default distance is L2; we create the collection with cosine space.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .embedders import BaseEmbedder
from .schemas import Chunk

logger = logging.getLogger(__name__)


class EmbeddingMismatchError(RuntimeError):
    """Raised when a collection built with one model is opened with another."""


@dataclass
class StoredRecord:
    chunk_id: str
    text: str
    metadata: dict[str, Any]


@dataclass
class DenseHit:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    distance: float          # Chroma cosine distance = 1 - cosine similarity
    embedding: np.ndarray    # the stored chunk vector d


class BaseVectorStore(ABC):
    """What the retriever needs from any vector database."""

    @property
    @abstractmethod
    def embedder(self) -> BaseEmbedder: ...

    @abstractmethod
    def add_chunks(self, chunks: Iterable[Chunk | dict]) -> int: ...

    @abstractmethod
    def search_by_vector(
        self, query_vector: np.ndarray, k: int, category: str | None = None
    ) -> list[DenseHit]: ...

    @abstractmethod
    def get_embeddings(self, ids: Sequence[str]) -> dict[str, np.ndarray]: ...

    @abstractmethod
    def all_records(self) -> list[StoredRecord]: ...

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def reset(self) -> None: ...

    def list_categories(self) -> list[str]:
        return sorted({str(r.metadata.get("category", "general")) for r in self.all_records()})


class ChromaVectorStore(BaseVectorStore):
    def __init__(
        self,
        embedder: BaseEmbedder,
        persist_dir: str | Path = "vector_db",
        collection_name: str = "nsbm_admin",
        client: Any = None,            # inject a client in tests; default is on-disk
        batch_size: int = 64,
    ):
        self._embedder = embedder
        self._name = collection_name
        self._batch_size = batch_size
        self.persist_dir = Path(persist_dir)
        if client is None:
            import chromadb
            from chromadb.config import Settings

            self.persist_dir.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(
                path=str(self.persist_dir), settings=Settings(anonymized_telemetry=False)
            )
        self._client = client
        self._collection = self._open_collection()

    # ------------------------------------------------------------------ setup
    @property
    def embedder(self) -> BaseEmbedder:
        return self._embedder

    @property
    def collection_name(self) -> str:
        return self._name

    def _existing_names(self) -> set[str]:
        # chromadb >= 0.6 returns Collection objects, older versions return names
        return {c if isinstance(c, str) else c.name for c in self._client.list_collections()}

    def _open_collection(self):
        if self._name in self._existing_names():
            # Open without passing metadata again (changing it on reopen is not allowed).
            collection = self._client.get_collection(name=self._name, embedding_function=None)
            stored_model = (collection.metadata or {}).get("embedding_model")
            if stored_model and stored_model != self._embedder.name:
                raise EmbeddingMismatchError(
                    f"Collection '{self._name}' was built with '{stored_model}' but the "
                    f"current embedder is '{self._embedder.name}'. Use another collection "
                    "name or rebuild it with --reset."
                )
            return collection
        logger.info("Creating Chroma collection '%s' (cosine space)", self._name)
        metadata = {
            "embedding_model": self._embedder.name,
            "description": "NSBM Academic Administration document chunks",
        }
        try:  # chromadb >= 1.0: index settings go in `configuration`
            return self._client.create_collection(
                name=self._name,
                embedding_function=None,  # we always supply our own vectors
                configuration={"hnsw": {"space": "cosine"}},
                metadata=metadata,
            )
        except TypeError:  # chromadb < 1.0 has no `configuration` argument
            return self._client.create_collection(
                name=self._name,
                embedding_function=None,
                metadata={**metadata, "hnsw:space": "cosine"},
            )

    # ------------------------------------------------------------------ write
    def add_chunks(self, chunks: Iterable[Chunk | dict]) -> int:
        """Embed and upsert chunks. Safe to run again: same chunk -> same id -> no duplicates."""
        unique: dict[str, Chunk] = {}
        for raw in chunks:
            chunk = Chunk.from_any(raw)
            unique.setdefault(chunk.chunk_id, chunk)
        records = list(unique.values())

        for start in range(0, len(records), self._batch_size):
            batch = records[start : start + self._batch_size]
            vectors = self._embedder.embed_documents([c.text for c in batch])
            self._collection.upsert(
                ids=[c.chunk_id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[{**c.to_metadata(), "embedding_model": self._embedder.name} for c in batch],
                embeddings=np.asarray(vectors, dtype=np.float32).tolist(),
            )
            logger.info("Upserted %d/%d chunks", min(start + len(batch), len(records)), len(records))
        return len(records)

    def reset(self) -> None:
        if self._name in self._existing_names():
            self._client.delete_collection(name=self._name)
        self._collection = self._open_collection()

    # ------------------------------------------------------------------- read
    def count(self) -> int:
        return int(self._collection.count())

    def search_by_vector(
        self, query_vector: np.ndarray, k: int, category: str | None = None
    ) -> list[DenseHit]:
        total = self.count()
        if total == 0 or k <= 0:
            return []
        result = self._collection.query(
            query_embeddings=[np.asarray(query_vector, dtype=np.float32).tolist()],
            n_results=min(k, total),
            where={"category": category} if category else None,
            include=["documents", "metadatas", "distances", "embeddings"],
        )
        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]
        embeddings = result["embeddings"][0]  # may be a numpy array: never test it with `if`
        return [
            DenseHit(
                chunk_id=ids[i],
                text=documents[i],
                metadata=dict(metadatas[i] or {}),
                distance=float(distances[i]),
                embedding=np.asarray(embeddings[i], dtype=np.float32),
            )
            for i in range(len(ids))
        ]

    def get_embeddings(self, ids: Sequence[str]) -> dict[str, np.ndarray]:
        if not ids:
            return {}
        result = self._collection.get(ids=list(ids), include=["embeddings"])
        return {
            cid: np.asarray(vec, dtype=np.float32)
            for cid, vec in zip(result["ids"], result["embeddings"])
        }

    def all_records(self) -> list[StoredRecord]:
        result = self._collection.get(include=["documents", "metadatas"])
        return [
            StoredRecord(chunk_id=cid, text=doc, metadata=dict(meta or {}))
            for cid, doc, meta in zip(result["ids"], result["documents"], result["metadatas"])
        ]
