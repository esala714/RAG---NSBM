"""Evaluation metrics, written as small pure functions so each one can be checked by hand.

Retrieval (no LLM needed), for each answerable question and its top-k chunks:

    Hit@k              = 1 if any retrieved chunk comes from an expected source document, else 0
    Reciprocal rank    = 1 / rank of the first chunk from an expected source (0 if none)
    Chunk Hit@k        = 1 if any retrieved chunk holds the answer, else 0
    Chunk RR           = 1 / rank of the first chunk that holds the answer (0 if none)
                         a chunk holds the answer when it contains at least 90% of the
                         content words of the ground-truth context passage
    Context Relevance  = (chunks containing at least one relevant term) / (chunks retrieved)
    Context Recall     = (words of the ground-truth context passage found in the chunks)
                         / (words of the passage)

Answers (need the agent and Gemini):

    Groundedness       = (answer sentences supported by the context) / (answer sentences checked)
                         a sentence is supported when at least 60% of its content words
                         also appear in the retrieved chunks
    Answer Relevance   = (expected keywords found in the answer) / (expected keywords)

Words are compared with Member 2's tokenizer (lower case, stop words removed, light
stemming), so "examinations" in an answer matches "examination" in a chunk.
"""
from __future__ import annotations

import re
from typing import Iterable, Sequence

from src.retrieval.sparse_index import tokenize

SUPPORT_THRESHOLD = 0.6     # share of a sentence's words that must appear in the context
ANSWER_CHUNK_COVERAGE = 0.9  # share of the passage's words one chunk must hold to count as the answer
MIN_SENTENCE_WORDS = 3      # shorter fragments ("Sources:", "[1]") are not checked

_CITATION_RE = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")      # [1] or [1, 2]
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _norm_source(name: str) -> str:
    return " ".join(str(name).lower().split())


# --------------------------------------------------------------------------- #
# Retrieval                                                                   #
# --------------------------------------------------------------------------- #
def hit_at_k(retrieved_sources: Sequence[str], expected_sources: Iterable[str]) -> float:
    expected = {_norm_source(s) for s in expected_sources}
    return 1.0 if any(_norm_source(s) in expected for s in retrieved_sources) else 0.0


def reciprocal_rank(retrieved_sources: Sequence[str], expected_sources: Iterable[str]) -> float:
    expected = {_norm_source(s) for s in expected_sources}
    for rank, source in enumerate(retrieved_sources, start=1):
        if _norm_source(source) in expected:
            return 1.0 / rank
    return 0.0


def context_recall(chunk_texts: Sequence[str], reference_context: str) -> float:
    """How much of the ground-truth passage that answers the question was retrieved (0 to 1)."""
    passage = set(tokenize(reference_context))
    if not passage:
        return 0.0
    retrieved = {w for text in chunk_texts for w in tokenize(text)}
    return len(passage & retrieved) / len(passage)


def passage_coverage(chunk_text: str, reference_context: str) -> float:
    """Share of the ground-truth passage's content words that are in this one chunk (0 to 1)."""
    passage = set(tokenize(reference_context))
    if not passage:
        return 0.0
    return len(passage & set(tokenize(chunk_text))) / len(passage)


def answer_chunk_rank(chunk_texts: Sequence[str], reference_context: str,
                      threshold: float = ANSWER_CHUNK_COVERAGE) -> int | None:
    """Rank of the first retrieved chunk that holds the answering passage (None if none does).

    Hit@k only asks whether a chunk comes from the right document, so a title page or another
    section of that document also counts. This asks whether the answer itself was retrieved.
    """
    for rank, text in enumerate(chunk_texts, start=1):
        if passage_coverage(text, reference_context) >= threshold:
            return rank
    return None


def context_relevance(chunk_texts: Sequence[str], relevant_terms: Sequence[str]) -> float:
    if not chunk_texts or not relevant_terms:
        return 0.0
    terms = [t.lower() for t in relevant_terms]
    relevant = sum(1 for text in chunk_texts if any(t in text.lower() for t in terms))
    return relevant / len(chunk_texts)


# --------------------------------------------------------------------------- #
# Answers                                                                     #
# --------------------------------------------------------------------------- #
def keyword_coverage(text: str, keywords: Sequence[str]) -> float:
    if not keywords:
        return 0.0
    lowered = text.lower()
    return sum(1 for k in keywords if k.lower() in lowered) / len(keywords)


def answer_relevance(answer: str, expected_keywords: Sequence[str]) -> float:
    return keyword_coverage(answer, expected_keywords)


def split_sentences(answer: str) -> list[str]:
    cleaned = _CITATION_RE.sub(" ", answer)
    cleaned = re.sub(r"[ \t]+([.,;:!?])", r"\1", cleaned)   # "start [1]." -> "start."
    return [s.strip(" -*\t") for s in _SENTENCE_RE.split(cleaned) if s.strip(" -*\t")]


def sentence_support(sentence: str, context_words: set[str]) -> float | None:
    """Share of the sentence's content words found in the context (None if too short to judge)."""
    words = tokenize(sentence)
    if len(words) < MIN_SENTENCE_WORDS:
        return None
    return sum(1 for w in words if w in context_words) / len(words)


def groundedness(answer: str, context_texts: Sequence[str],
                 threshold: float = SUPPORT_THRESHOLD) -> tuple[float | None, list[dict]]:
    """Returns (score, per-sentence details). The score is None when no sentence could be checked."""
    context_words = {w for text in context_texts for w in tokenize(text)}
    details = []
    for sentence in split_sentences(answer):
        support = sentence_support(sentence, context_words)
        if support is None:
            continue
        details.append({
            "sentence": sentence,
            "support": round(support, 3),
            "supported": support >= threshold,
        })
    if not details:
        return None, details
    return sum(d["supported"] for d in details) / len(details), details


def mean(values: Iterable[float | None]) -> float | None:
    kept = [v for v in values if v is not None]
    return sum(kept) / len(kept) if kept else None
