from abc import ABC, abstractmethod

from src.core.models import RetrievedChunk


class BaseRetriever(ABC):
    """Rule: every retriever must have a search() method.
    Member 2's vector store class must inherit from this."""

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Return the top_k most relevant chunks for the query."""
        ...


class BaseLLMClient(ABC):
    """Rule: every LLM client must have a generate() method.
    Gemini today; could be another LLM later."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Send the prompt to the LLM and return its text reply."""
        ...