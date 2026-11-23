"""TF-IDF vector space model, built from scratch with NumPy (no scikit-learn).

Every chunk becomes a vector with one dimension per vocabulary word:

    tf(t, d)   = 1 + log(count of t in d)          (0 if t is not in d)
    idf(t)     = log((1 + N) / (1 + df(t))) + 1    (rare words weigh more)
    w(t, d)    = tf(t, d) * idf(t)
    d_hat      = w / ||w||                          (L2 normalisation)
    score(q,d) = q_hat . d_hat = cosine similarity

Because both vectors are unit length, the cosine is literally the sum of
q_i * d_i over the words they share - which is what the trace prints.

Why have it next to the embeddings? Dense vectors match meaning
("leave" ~ "absence"), TF-IDF matches exact terms ("MBA", "C-", "2.0").
Fusing both is the hybrid retriever required for the higher grade bands.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Collection, Sequence

import numpy as np

# common English words + question filler words ("need", "many", "tell") that carry no policy meaning
STOPWORDS = frozenset(
    """a an the and or of to in on for at by with from is are was were be been being as
    that this these those it its into than then there their they he she his her them
    which who whom whose what when where why how all any each other such no not nor only
    own same so too very can will shall should would may might must do does did done
    have has had having i you your yours we our us me my also if per via
    need get many much about tell know please want""".split()
)

# words, plus decimals such as "2.0" or "3.70" (GPA values matter here)
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def _light_stem(token: str) -> str:
    """Tiny suffix rule so 'examinations' and 'examination' share one dimension."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    tokens = []
    for tok in _TOKEN_RE.findall(text.lower()):
        if tok in STOPWORDS or (len(tok) < 2 and not tok.isdigit()):
            continue
        tokens.append(_light_stem(tok))
    return tokens


@dataclass
class SparseQuery:
    raw: np.ndarray                 # TF-IDF weights before normalisation
    unit: np.ndarray                # raw / ||raw||
    raw_norm: float
    weights: list[tuple[str, float]]
    unknown_terms: list[str]


@dataclass
class SparseHit:
    chunk_id: str
    cosine: float
    rank: int


class TfidfIndex:
    def __init__(self) -> None:
        self.vocabulary: dict[str, int] = {}
        self.terms: list[str] = []
        self.idf = np.zeros(0)
        self.matrix = np.zeros((0, 0), dtype=np.float32)  # N chunks x V terms, rows unit length
        self.ids: list[str] = []
        self._row: dict[str, int] = {}

    @property
    def n_docs(self) -> int:
        return len(self.ids)

    @property
    def vocabulary_size(self) -> int:
        return len(self.terms)

    def fit(self, ids: Sequence[str], texts: Sequence[str]) -> "TfidfIndex":
        docs_tokens = [tokenize(t) for t in texts]
        self.terms = sorted({tok for toks in docs_tokens for tok in toks})
        self.vocabulary = {term: i for i, term in enumerate(self.terms)}
        self.ids = list(ids)
        self._row = {cid: i for i, cid in enumerate(self.ids)}

        n, v = len(self.ids), len(self.terms)
        counts = np.zeros((n, v), dtype=np.float32)
        for row, toks in enumerate(docs_tokens):
            for tok, c in Counter(toks).items():
                counts[row, self.vocabulary[tok]] = c

        df = (counts > 0).sum(axis=0)                       # document frequency
        self.idf = np.log((1 + n) / (1 + df)) + 1.0          # smoothed idf
        tf = np.where(counts > 0, 1.0 + np.log(np.maximum(counts, 1.0)), 0.0)
        weights = tf * self.idf
        norms = np.linalg.norm(weights, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.matrix = (weights / norms).astype(np.float32)
        return self

    def vectorize(self, text: str) -> SparseQuery:
        raw = np.zeros(self.vocabulary_size, dtype=np.float64)
        unknown: list[str] = []
        for tok, c in Counter(tokenize(text)).items():
            idx = self.vocabulary.get(tok)
            if idx is None:
                unknown.append(tok)
                continue
            raw[idx] = (1.0 + math.log(c)) * self.idf[idx]
        norm = float(np.linalg.norm(raw))
        unit = raw / norm if norm > 0 else raw
        weights = sorted(
            ((self.terms[i], float(raw[i])) for i in np.flatnonzero(raw)), key=lambda x: -x[1]
        )
        return SparseQuery(raw=raw, unit=unit, raw_norm=norm, weights=weights, unknown_terms=unknown)

    def search(
        self, query: SparseQuery, k: int, allowed_ids: Collection[str] | None = None
    ) -> list[SparseHit]:
        if self.n_docs == 0 or query.raw_norm == 0:
            return []
        scores = self.matrix @ query.unit.astype(np.float32)   # one dot product per chunk
        if allowed_ids is not None:
            mask = np.array([cid in allowed_ids for cid in self.ids])
            scores = np.where(mask, scores, -1.0)
        order = np.argsort(-scores)
        hits = []
        for idx in order[:k]:
            if scores[idx] <= 0:
                break
            hits.append(SparseHit(self.ids[idx], float(scores[idx]), len(hits) + 1))
        return hits

    def score(self, query: SparseQuery, chunk_id: str) -> float:
        row = self._row.get(chunk_id)
        if row is None or query.raw_norm == 0:
            return 0.0
        return float(self.matrix[row] @ query.unit)

    def term_contributions(
        self, query: SparseQuery, chunk_id: str, top: int = 5
    ) -> list[tuple[str, float]]:
        """The q_i * d_i products that add up to the cosine, largest first."""
        row = self._row.get(chunk_id)
        if row is None or query.raw_norm == 0:
            return []
        products = self.matrix[row] * query.unit
        shared = np.flatnonzero(products > 0)
        ranked = sorted(((self.terms[i], float(products[i])) for i in shared), key=lambda x: -x[1])
        return ranked[:top]
