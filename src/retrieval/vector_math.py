"""Session-1 text mathematics, written out explicitly.

Both retrievers (dense embeddings and sparse TF-IDF) rank chunks with the
same formula, so this is the one place to point at in the viva:

    magnitude      ||v||      = sqrt(v_1^2 + v_2^2 + ... + v_n^2)
    dot product    a . b      = a_1*b_1 + a_2*b_2 + ... + a_n*b_n
    normalisation  v_hat      = v / ||v||            (unit length, ||v_hat|| = 1)
    cosine         cos(a, b)  = (a . b) / (||a|| * ||b||) = a_hat . b_hat
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _as_vector(v) -> np.ndarray:
    return np.asarray(v, dtype=np.float64).ravel()


def magnitude(v) -> float:
    v = _as_vector(v)
    return float(np.sqrt(np.dot(v, v)))


def dot(a, b) -> float:
    a, b = _as_vector(a), _as_vector(b)
    if a.shape != b.shape:
        raise ValueError(f"Vectors have different dimensions: {a.shape} vs {b.shape}")
    return float(np.dot(a, b))


def normalize(v) -> np.ndarray:
    v = _as_vector(v)
    n = magnitude(v)
    return v / n if n > 0 else v


def cosine(a, b) -> float:
    na, nb = magnitude(a), magnitude(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot(a, b) / (na * nb)


@dataclass(frozen=True)
class CosineExplanation:
    dot: float
    norm_a: float
    norm_b: float
    cosine: float


def explain_cosine(a, b) -> CosineExplanation:
    """Return every intermediate value of the cosine calculation."""
    if b is None:
        return CosineExplanation(0.0, magnitude(a), 0.0, 0.0)
    d = dot(a, b)
    na, nb = magnitude(a), magnitude(b)
    cos = d / (na * nb) if na and nb else 0.0
    return CosineExplanation(dot=d, norm_a=na, norm_b=nb, cosine=cos)
