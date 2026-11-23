"""Member 2 - Embeddings and Vector Database (retrieval layer).

Typical use:
    from src.retrieval import build_retriever
    retriever = build_retriever()
    result = retriever.retrieve("What GPA do I need for a First Class?", k=4)
    print(result.explain())
"""
from .embedders import BaseEmbedder, GeminiEmbedder, MiniLMEmbedder
from .factory import build_embedder, build_retriever, build_store
from .retriever import HybridRetriever
from .schemas import Chunk, RetrievalResult, RetrievedChunk, ScoreBreakdown
from .sparse_index import TfidfIndex
from .vector_store import BaseVectorStore, ChromaVectorStore, EmbeddingMismatchError

__all__ = [
    "BaseEmbedder", "MiniLMEmbedder", "GeminiEmbedder",
    "BaseVectorStore", "ChromaVectorStore", "EmbeddingMismatchError",
    "TfidfIndex", "HybridRetriever",
    "Chunk", "RetrievalResult", "RetrievedChunk", "ScoreBreakdown",
    "build_embedder", "build_store", "build_retriever",
]
