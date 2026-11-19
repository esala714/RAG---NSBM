"""Splitting clean text into overlapping word windows (chunks)."""
from __future__ import annotations

# MiniLM (Member 2's embedding model) reads only the first 256 word-pieces of a chunk,
# about 192 English words. 180-word chunks fit inside that limit, so dense search sees
# every word; the 40-word overlap keeps a sentence cut at a chunk edge whole in the next chunk.
CHUNK_SIZE = 180
CHUNK_OVERLAP = 40


class WordChunker:
    """Sliding window over words: each window starts (chunk_size - overlap) words after the last."""

    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1 word.")
        if overlap < 0:
            raise ValueError("overlap cannot be negative.")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str) -> list[str]:
        words = text.split()
        if not words:
            return []
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunks.append(" ".join(words[start:end]))
            if end == len(words):
                break
            # always move forward by at least one word, so overlap >= chunk_size cannot loop forever
            start = max(end - self.overlap, start + 1)
        return chunks
