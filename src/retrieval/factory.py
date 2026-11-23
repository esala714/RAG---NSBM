"""Composition root: the only place that decides WHICH concrete classes are used.

Everything else receives its dependencies through constructors (dependency
injection), so Member 4's app.py just calls ``build_retriever()``.

Settings come from environment variables (see env.example):
    EMBEDDING_BACKEND   minilm (default, free, offline) | gemini
    CHROMA_DIR          folder for the persistent database (default: vector_db)
    CHROMA_COLLECTION   collection name (default: nsbm_admin_<backend>)
    RETRIEVAL_MODE      hybrid (default) | dense | sparse
    GEMINI_API_KEY      only needed when EMBEDDING_BACKEND=gemini
"""
from __future__ import annotations

import os

from .embedders import BaseEmbedder, GeminiEmbedder, MiniLMEmbedder
from .retriever import HybridRetriever
from .vector_store import ChromaVectorStore

try:  # optional: read a local .env file if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def build_embedder(backend: str | None = None) -> BaseEmbedder:
    backend = (backend or os.getenv("EMBEDDING_BACKEND", "minilm")).lower()
    if backend == "minilm":
        return MiniLMEmbedder()
    if backend == "gemini":
        from google import genai

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("EMBEDDING_BACKEND=gemini needs GEMINI_API_KEY to be set.")
        dims = int(os.getenv("GEMINI_EMBED_DIM", "768"))
        return GeminiEmbedder(genai.Client(api_key=api_key), dimensions=dims)
    raise ValueError(f"Unknown EMBEDDING_BACKEND '{backend}' (use 'minilm' or 'gemini').")


def build_store(backend: str | None = None, persist_dir: str | None = None,
                collection_name: str | None = None) -> ChromaVectorStore:
    backend = (backend or os.getenv("EMBEDDING_BACKEND", "minilm")).lower()
    return ChromaVectorStore(
        embedder=build_embedder(backend),
        persist_dir=persist_dir or os.getenv("CHROMA_DIR", "vector_db"),
        # one collection per model, so switching models never mixes vector sizes
        collection_name=collection_name or os.getenv("CHROMA_COLLECTION", f"nsbm_admin_{backend}"),
    )


def build_retriever(backend: str | None = None, persist_dir: str | None = None,
                    collection_name: str | None = None, mode: str | None = None) -> HybridRetriever:
    store = build_store(backend, persist_dir, collection_name)
    return HybridRetriever(store, mode=mode or os.getenv("RETRIEVAL_MODE", "hybrid"))
