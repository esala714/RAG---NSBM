from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievedChunk:
    """One piece of an NSBM academic document found by the search."""
    text: str                # the chunk's text
    source: str              # file name, e.g. "exam_regulations.pdf"
    page: int | None         # page number (None if unknown)
    score: float             # similarity: higher = more relevant (0 to 1)
    extra: dict[str, Any] = field(default_factory=dict)  # math details, e.g. dot product


@dataclass
class AgentStep:
    """One line in the Agent's trace (what it did and why)."""
    phase: str               # "observation", "reasoning", "action" or "persistence"
    detail: str              # a short description of what happened


@dataclass
class RAGResponse:
    """Everything the Agent returns after answering a question."""
    question: str
    answer: str
    sources: list[RetrievedChunk]
    trace: list[AgentStep] = field(default_factory=list)
    answerable: bool = True  # False if the documents could not answer it