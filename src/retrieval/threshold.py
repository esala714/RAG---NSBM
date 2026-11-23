"""Choose the agent's refusal threshold (min_score) from retrieval scores.

The agent answers a question only when its best retrieved chunk has a score of
at least min_score (the dense cosine from HybridRetriever.search()), and refuses
otherwise. Given the best score of questions the documents CAN answer and of
questions they CANNOT, a threshold t is judged by its balanced accuracy

    answered(t) = share of answerable questions with score >= t
    refused(t)  = share of unanswerable questions with score <  t
    balanced(t) = (answered(t) + refused(t)) / 2

Both groups count equally however many questions each has. Plain accuracy
would not do: with 14 answerable and 10 unanswerable questions, answering
everything already scores 14/24, so it can win over a useful threshold.

balanced(t) only changes when t passes one of the scores, so it is enough to
try one t in each gap between two neighbouring scores (and below the lowest and
above the highest one). Among the gaps with the best balanced accuracy, the
widest gap is kept and its middle is returned: that threshold is as far as
possible from the nearest score on either side (largest margin), so small
changes in the scores are least likely to flip a decision.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ThresholdChoice:
    threshold: float      # answer when best score >= threshold
    answered: int         # answerable questions that pass the threshold
    refused: int          # unanswerable questions that fall below it
    n_answerable: int
    n_unanswerable: int
    margin: float         # distance from the threshold to the nearest score

    @property
    def correct(self) -> int:
        return self.answered + self.refused

    @property
    def accuracy(self) -> float:
        return self.correct / (self.n_answerable + self.n_unanswerable)

    @property
    def balanced_accuracy(self) -> float:
        return balanced_accuracy(self.answered, self.refused, self.n_answerable, self.n_unanswerable)


def balanced_accuracy(answered: int, refused: int, n_answerable: int, n_unanswerable: int) -> float:
    return (answered / n_answerable + refused / n_unanswerable) / 2


def outcome(threshold: float, answerable: Iterable[float], unanswerable: Iterable[float]) -> tuple[int, int]:
    """(answerable questions answered, unanswerable questions refused) at this threshold."""
    answered = sum(1 for s in answerable if s >= threshold)
    refused = sum(1 for s in unanswerable if s < threshold)
    return answered, refused


def sweep(answerable: Sequence[float], unanswerable: Sequence[float],
          thresholds: Iterable[float]) -> list[tuple[float, int, int]]:
    """(threshold, answered, refused) for each threshold, for a table in the report."""
    return [(t, *outcome(t, answerable, unanswerable)) for t in thresholds]


def choose_threshold(answerable: Sequence[float], unanswerable: Sequence[float]) -> ThresholdChoice:
    if not answerable or not unanswerable:
        raise ValueError("Need at least one answerable and one unanswerable score.")
    points = sorted({min(1.0, max(0.0, float(s))) for s in (*answerable, *unanswerable)})
    bounds = [0.0, *points, 1.0]

    best: tuple[tuple[float, float], ThresholdChoice] | None = None
    for low, high in zip(bounds, bounds[1:]):
        if high <= low:          # a score of exactly 0 or 1 makes an empty gap
            continue
        t = (low + high) / 2
        answered, refused = outcome(t, answerable, unanswerable)
        key = (balanced_accuracy(answered, refused, len(answerable), len(unanswerable)), high - low)
        if best is None or key > best[0]:
            choice = ThresholdChoice(t, answered, refused, len(answerable), len(unanswerable), (high - low) / 2)
            best = (key, choice)
    assert best is not None
    return best[1]
