"""Tests for choosing the agent's refusal threshold (no Chroma or model needed)."""
import pytest

from src.retrieval.threshold import choose_threshold, outcome, sweep


def test_clean_separation_puts_the_threshold_in_the_middle_of_the_gap():
    choice = choose_threshold(answerable=[0.60, 0.55, 0.50], unanswerable=[0.20, 0.30])
    assert choice.threshold == pytest.approx(0.40)      # middle of the gap 0.30 .. 0.50
    assert choice.margin == pytest.approx(0.10)
    assert (choice.answered, choice.refused, choice.accuracy) == (3, 2, 1.0)


def test_overlap_keeps_the_widest_of_the_best_gaps():
    # Two gaps give 3 of 4 correct: 0.10..0.25 (width 0.15) and 0.30..0.60 (width 0.30).
    choice = choose_threshold(answerable=[0.60, 0.25], unanswerable=[0.30, 0.10])
    assert choice.correct == 3
    assert choice.threshold == pytest.approx(0.45)
    assert (choice.answered, choice.refused) == (1, 2)


def test_answering_everything_does_not_win_when_one_group_is_larger():
    # Answering all 7 gets 6 right (plain accuracy 0.86) but refuses nothing.
    choice = choose_threshold(answerable=[0.9, 0.8, 0.7, 0.6, 0.45, 0.4], unanswerable=[0.5])
    assert choice.threshold == pytest.approx(0.55)
    assert (choice.answered, choice.refused) == (4, 1)
    assert choice.balanced_accuracy == pytest.approx((4 / 6 + 1) / 2)


def test_a_score_equal_to_the_threshold_is_answered():
    assert outcome(0.35, answerable=[0.35], unanswerable=[0.35]) == (1, 0)


def test_sweep_counts_each_threshold():
    rows = sweep([0.6, 0.4], [0.3], [0.2, 0.5])
    assert rows == [(0.2, 2, 0), (0.5, 1, 1)]


def test_both_groups_are_needed():
    with pytest.raises(ValueError):
        choose_threshold([0.5], [])
