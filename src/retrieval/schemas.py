"""Data contracts for the retrieval layer.

INPUT  (from Member 1)  -> ``Chunk``            one cleaned piece of text + where it came from
OUTPUT (to Member 3/4)  -> ``RetrievalResult``  the best chunks + the maths behind every score

Keeping these in one file means the other members only need to agree on
these classes, not on how the retrieval works inside.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

# Member 1 may name fields slightly differently - we accept the common spellings.
_TEXT_KEYS = ("text", "content", "page_content", "chunk_text")
_SOURCE_KEYS = ("source", "document", "document_name", "doc_name", "file_name", "filename")
_CATEGORY_KEYS = ("category", "folder", "doc_type")
_PAGE_KEYS = ("page", "page_number", "page_no")
_INDEX_KEYS = ("chunk_index", "index", "chunk_no")


def _first(data: Mapping[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    for key in keys:
        value = data.get(key)
        if value is not None and value != "":
            return value
    return default


# --------------------------------------------------------------------------- #
# INPUT                                                                        #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Chunk:
    """One chunk produced by Member 1's document processor."""

    text: str
    source: str               # document file name, e.g. "NSBM_Leave_Policy.pdf"
    category: str = "general"  # folder name, e.g. "Exam", "GPA", "leave_policy"
    page: int = 1             # 1-based page number (DOCX files have no pages -> 1)
    chunk_index: int = 0      # position of the chunk inside its page

    @property
    def chunk_id(self) -> str:
        """Deterministic id: the same chunk always gets the same id,
        so re-running the indexer updates rows instead of duplicating them."""
        raw = f"{self.source}|{self.page}|{self.chunk_index}|{self.text}"
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
        stem = re.sub(r"[^A-Za-z0-9]+", "_", self.source.rsplit(".", 1)[0]).strip("_")[:40]
        return f"{stem}-p{self.page}-c{self.chunk_index}-{digest}"

    def to_metadata(self) -> dict[str, Any]:
        """Chroma metadata values must be str / int / float / bool (no None)."""
        return {
            "source": self.source,
            "category": self.category,
            "page": int(self.page),
            "chunk_index": int(self.chunk_index),
            "char_count": len(self.text),
        }

    @classmethod
    def from_any(cls, obj: Any) -> "Chunk":
        """Accept a Chunk, a dict (flat or with a nested 'metadata' dict),
        or any object with .text/.page_content and .metadata attributes."""
        if isinstance(obj, Chunk):
            return obj
        if isinstance(obj, Mapping):
            data = dict(obj)
        else:
            data = dict(getattr(obj, "__dict__", {}))
            for attr in _TEXT_KEYS + ("metadata",):
                if hasattr(obj, attr):
                    data[attr] = getattr(obj, attr)

        nested = data.pop("metadata", None) or {}
        merged = {**nested, **data}

        text = _first(merged, _TEXT_KEYS)
        if text is None or not str(text).strip():
            raise ValueError(f"Chunk has no text. Got keys: {sorted(merged)}")
        return cls(
            text=str(text).strip(),
            source=str(_first(merged, _SOURCE_KEYS, "unknown")),
            category=str(_first(merged, _CATEGORY_KEYS, "general")),
            page=int(_first(merged, _PAGE_KEYS, 1)),
            chunk_index=int(_first(merged, _INDEX_KEYS, 0)),
        )


# --------------------------------------------------------------------------- #
# OUTPUT                                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class ScoreBreakdown:
    """Every number that decided where a chunk was ranked."""

    # dense (embedding) side: cos = (q . d) / (||q|| * ||d||)
    dot: float
    query_norm: float
    chunk_norm: float
    dense_cosine: float
    chroma_distance: float | None  # Chroma's own cosine distance (= 1 - cos), for cross-checking
    # sparse (TF-IDF) side
    sparse_cosine: float
    sparse_terms: list[tuple[str, float]]  # shared terms and their q_i * d_i contribution
    # fusion
    dense_rank: int | None
    sparse_rank: int | None
    rrf_score: float
    fusion_formula: str


@dataclass
class RetrievedChunk:
    rank: int
    chunk_id: str
    text: str
    source: str
    category: str
    page: int
    scores: ScoreBreakdown

    @property
    def citation(self) -> str:
        return f"{self.source}, page {self.page}"


@dataclass
class RetrievalResult:
    query: str
    mode: str
    chunks: list[RetrievedChunk]
    embedding_model: str
    query_dim: int
    query_norm: float
    query_terms: list[tuple[str, float]]  # TF-IDF weights of the query before normalisation
    query_sparse_norm: float
    unknown_terms: list[str]              # query words that never occur in the corpus
    rrf_k: int
    elapsed_ms: float
    category_filter: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    # ---- helpers for Member 3 (agent / prompt) -------------------------------
    @property
    def max_dense_cosine(self) -> float:
        """Best semantic match. Member 3 can refuse to answer when this is low
        (calibrate the threshold with the unanswerable ground-truth questions)."""
        return max((c.scores.dense_cosine for c in self.chunks), default=0.0)

    @property
    def sources(self) -> list[str]:
        return sorted({c.citation for c in self.chunks})

    def to_context(self) -> str:
        """Numbered passages with citations, ready to paste into the LLM prompt."""
        blocks = [
            f"[{c.rank}] Source: {c.source} (page {c.page}, category {c.category})\n{c.text}"
            for c in self.chunks
        ]
        return "\n\n".join(blocks)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    # ---- explainability printout (for the video / viva) -----------------------
    def explain(self, preview_chars: int = 220) -> str:
        """ASCII-only so it prints safely in Colab and on Windows terminals."""
        terms = ", ".join(f"{t}={w:.2f}" for t, w in self.query_terms) or "(none)"
        lines = [
            f'QUERY : "{self.query}"',
            f"MODE  : {self.mode} (RRF k={self.rrf_k}) | category filter: "
            f"{self.category_filter or 'none'} | {self.elapsed_ms:.0f} ms",
            f"DENSE : model={self.embedding_model}, dim={self.query_dim}, ||q|| = {self.query_norm:.4f}",
            f"SPARSE: query TF-IDF weights before normalisation: {terms}",
            f"        ||q_raw|| = {self.query_sparse_norm:.4f}  ->  q_hat = q_raw / {self.query_sparse_norm:.4f}"
            "  (unit length)",
        ]
        if self.unknown_terms:
            lines.append(f"        not in vocabulary (ignored): {', '.join(self.unknown_terms)}")
        lines.append("-" * 78)
        if not self.chunks:
            lines.append("No chunks retrieved.")
        for c in self.chunks:
            s = c.scores
            chroma = ""
            if s.chroma_distance is not None:
                chroma = f"   [Chroma distance {s.chroma_distance:.4f} -> 1 - dist = {1 - s.chroma_distance:.4f}]"
            shared = ", ".join(f"{t}:{w:.3f}" for t, w in s.sparse_terms) or "no shared terms"
            preview = " ".join(c.text.split())
            if len(preview) > preview_chars:
                preview = preview[:preview_chars].rstrip() + " ..."
            lines += [
                f"#{c.rank}  {c.citation}  [{c.category}]  id={c.chunk_id}",
                f"    dense : q.d = {s.dot:.4f}   ||q|| = {s.query_norm:.4f}   ||d|| = {s.chunk_norm:.4f}",
                f"            cos = q.d / (||q|| x ||d||) = {s.dense_cosine:.4f}{chroma}",
                f"    sparse: TF-IDF cos = {s.sparse_cosine:.4f}   shared terms (q_i x d_i): {shared}",
                f"    fusion: dense rank {s.dense_rank or '-'}, sparse rank {s.sparse_rank or '-'}"
                f"  ->  RRF = {s.fusion_formula} = {s.rrf_score:.5f}",
                f'    text  : "{preview}"',
                "",
            ]
        return "\n".join(lines)
