"""Offline baseline retriever for the evaluation, following Member 3's BaseRetriever interface.

TfidfRetriever  Member 2's TF-IDF index (NumPy only) built over Member 1's
                data/processed/chunks.jsonl. No Chroma, no model download and no API key,
                so the retrieval metrics can be reproduced on any laptop.

The full system needs no wrapper: Member 2's HybridRetriever implements BaseRetriever
itself, so the evaluation calls its search() exactly as the agent does.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.core.interfaces import BaseRetriever
from src.core.models import RetrievedChunk
from src.retrieval.schemas import Chunk
from src.retrieval.sparse_index import TfidfIndex


def load_chunks(path: str | Path) -> list[Chunk]:
    """Read Member 1's JSONL output (one chunk per line) into Member 2's Chunk objects."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Create it first with:\n"
            "  python -m src.ingestion.document_processor --input data/raw "
            "--output data/processed/chunks.jsonl"
        )
    with path.open(encoding="utf-8") as file:
        return [Chunk.from_any(json.loads(line)) for line in file if line.strip()]


class TfidfRetriever(BaseRetriever):
    name = "tfidf"

    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("TfidfRetriever needs at least one chunk.")
        self._chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self._index = TfidfIndex().fit(list(self._chunks), [c.text for c in self._chunks.values()])

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "TfidfRetriever":
        return cls(load_chunks(path))

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        hits = self._index.search(self._index.vectorize(query), top_k)
        results = []
        for hit in hits:
            chunk = self._chunks[hit.chunk_id]
            results.append(RetrievedChunk(
                chunk.text, chunk.source, chunk.page if chunk.page > 0 else None, hit.cosine,
                extra={"rank": hit.rank, "category": chunk.category, "chunk_id": hit.chunk_id,
                       "score_type": "tfidf_cosine"},
            ))
        return results
