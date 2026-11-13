"""Embedding models behind one interface (Strategy pattern).

The vector store receives a ``BaseEmbedder`` through its constructor
(dependency injection), so swapping MiniLM for Gemini is a one-line change
and never touches the store or retriever code.
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Sequence

import numpy as np

logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """Turns text into dense vectors."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stored with the collection so we never mix vectors from two models."""

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Return an array of shape (len(texts), dim)."""

    def embed_query(self, text: str) -> np.ndarray:
        """Return a 1-D vector. Override when the model treats queries differently."""
        return self.embed_documents([text])[0]


class MiniLMEmbedder(BaseEmbedder):
    """all-MiniLM-L6-v2 (384 dimensions), run locally by Chroma's built-in ONNX runtime.

    This is the same model Chroma uses when you call ``collection.query(query_texts=...)``,
    but we call it ourselves so we can see (and print) the actual vectors.
    Free, offline after the first download, no API key.
    """

    def __init__(self, embedding_function: Any = None):
        if embedding_function is None:
            from chromadb.utils import embedding_functions

            embedding_function = embedding_functions.DefaultEmbeddingFunction()
        self._ef = embedding_function

    @property
    def name(self) -> str:
        return "all-MiniLM-L6-v2"

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        return np.asarray(self._ef(list(texts)), dtype=np.float32)


class GeminiEmbedder(BaseEmbedder):
    """Google ``gemini-embedding-001`` through the google-genai client.

    Uses different task types for documents and queries (asymmetric retrieval).
    Needs an API key and is rate-limited on the free tier, so it is optional.
    """

    def __init__(
        self,
        client: Any,                       # an already-built genai.Client (injected)
        model: str = "gemini-embedding-001",
        dimensions: int = 768,
        batch_size: int = 50,
        max_retries: int = 5,
    ):
        self._client = client
        self._model = model
        self._dimensions = dimensions
        self._batch_size = batch_size
        self._max_retries = max_retries

    @property
    def name(self) -> str:
        return f"{self._model}-{self._dimensions}d"

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text], "RETRIEVAL_QUERY")[0]

    def _embed(self, texts: Sequence[str], task_type: str) -> np.ndarray:
        from google.genai import types

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            config = types.EmbedContentConfig(
                task_type=task_type, output_dimensionality=self._dimensions
            )
            for attempt in range(self._max_retries):
                try:
                    response = self._client.models.embed_content(
                        model=self._model, contents=batch, config=config
                    )
                    vectors.extend(e.values for e in response.embeddings)
                    break
                except Exception as exc:  # free tier returns 429 RESOURCE_EXHAUSTED
                    if attempt == self._max_retries - 1 or "429" not in str(exc):
                        raise
                    wait = 2 ** attempt * 5
                    logger.warning("Gemini rate limit hit, retrying in %ss", wait)
                    time.sleep(wait)
        return np.asarray(vectors, dtype=np.float32).reshape(len(vectors), self._dimensions)
