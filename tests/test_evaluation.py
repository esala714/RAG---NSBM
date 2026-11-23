"""Tests for the evaluation module.   Run:  python -m pytest -q tests/test_evaluation.py

Everything runs offline: no Chroma, no Gemini, no API key.
"""
import json
from pathlib import Path

import pytest

from src.core.models import AgentStep, RAGResponse, RetrievedChunk
from src.evaluation.evaluate import evaluate_agent, evaluate_retrieval, load_ground_truth
from src.evaluation.metrics import (
    answer_chunk_rank,
    answer_relevance,
    context_recall,
    context_relevance,
    groundedness,
    hit_at_k,
    reciprocal_rank,
    split_sentences,
)
from src.evaluation.pacing import PacedLLM, retry_delay
from src.evaluation.retrievers import TfidfRetriever
from src.retrieval.schemas import Chunk
from tests.fake_retriever import FakeRetriever

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_FILE = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"


# ---- metrics ----------------------------------------------------------------
def test_hit_and_reciprocal_rank():
    retrieved = ["a.pdf", "exam.pdf", "exam.pdf"]

    assert hit_at_k(retrieved, ["Exam.pdf"]) == 1.0
    assert reciprocal_rank(retrieved, ["exam.pdf"]) == 0.5
    assert hit_at_k(retrieved, ["other.pdf"]) == 0.0
    assert reciprocal_rank(retrieved, ["other.pdf"]) == 0.0


def test_context_relevance_counts_chunks_with_a_relevant_term():
    chunks = ["Admitted up to thirty minutes late", "Canteen opening hours", "No mobile phones"]

    assert context_relevance(chunks, ["thirty minutes", "mobile"]) == pytest.approx(2 / 3)
    assert context_relevance([], ["thirty minutes"]) == 0.0


def test_context_recall_is_the_share_of_the_passage_retrieved():
    passage = "Candidates may be admitted thirty minutes late"

    assert context_recall([passage], passage) == 1.0
    assert context_recall(["Canteen opening hours"], passage) == 0.0
    assert context_recall(["admitted thirty minutes"], "admitted thirty minutes late") == 0.75


def test_answer_chunk_rank_needs_the_passage_not_only_the_document():
    passage = "Poor internet connectivity will not be approved as an extenuating circumstance"
    title_page = "EXTENUATING CIRCUMSTANCES GUIDELINES FOR EXAMINATIONS"
    answer = "Technical issues: the following problems will not be approved. " + passage

    assert answer_chunk_rank([title_page, answer], passage) == 2     # Hit@k would count rank 1
    assert answer_chunk_rank([title_page], passage) is None
    assert answer_chunk_rank(["internet connectivity"], passage) is None   # under 90% of the passage


def test_answer_relevance_is_keyword_coverage():
    answer = "You can enter up to thirty minutes after the start."

    assert answer_relevance(answer, ["thirty minutes", "start", "invigilator"]) == pytest.approx(2 / 3)


def test_split_sentences_removes_citations():
    assert split_sentences("No. Phones are not permitted [1].\nSources: [1, 2]") == [
        "No.", "Phones are not permitted.", "Sources:"]


def test_groundedness_flags_unsupported_sentence():
    context = ["Candidates shall be admitted to the examination room up to thirty minutes "
               "after the official start of the examination."]
    answer = ("Candidates are admitted up to thirty minutes after the start of the examination [1]. "
              "Late candidates receive a free coffee voucher from the canteen.")

    score, details = groundedness(answer, context)

    assert score == 0.5
    assert [d["supported"] for d in details] == [True, False]


def test_groundedness_is_none_when_nothing_can_be_checked():
    score, details = groundedness("Yes [1].", ["any context"])

    assert score is None
    assert details == []


# ---- retrieval evaluation ---------------------------------------------------
QUESTIONS = [
    {"id": "t1", "question": "What attendance do I need for the final exam?", "answerable": True,
     "expected_sources": ["exam_regulations_SAMPLE.pdf"], "relevant_terms": ["attendance"],
     "expected_keywords": ["80%"]},
    {"id": "t2", "question": "What is on the canteen menu?", "answerable": False,
     "expected_sources": [], "relevant_terms": [], "expected_keywords": []},
]


def test_evaluate_retrieval_with_fake_retriever():
    result = evaluate_retrieval(FakeRetriever(), QUESTIONS, k=2)
    summary, rows = result["summary"], result["questions"]

    assert summary["hit_rate"] == 1.0
    assert summary["mrr"] == 1.0
    assert summary["context_relevance"] == 0.5          # 1 of the 2 sample chunks mentions attendance
    assert rows[0]["top_score"] == 0.81
    assert "hit" not in rows[1]                          # unanswerable: no retrieval score
    assert summary["highest_top_score_unanswerable"] == 0.81
    assert rows[0]["chunk_hit"] is None                  # no context passage: no chunk-level score


def test_evaluate_retrieval_reports_chunk_level_hits():
    question = dict(QUESTIONS[0], context="at least 80% attendance in a module to be eligible to sit "
                                          "its final examination")
    result = evaluate_retrieval(FakeRetriever(), [question], k=2)

    assert result["questions"][0]["answer_chunk_rank"] == 1
    assert result["summary"]["chunk_hits"] == 1
    assert result["summary"]["chunk_mrr"] == 1.0


def test_tfidf_retriever_returns_core_chunks():
    chunks = [
        Chunk("Candidates may enter the examination room up to thirty minutes late.", "exam.pdf", "examinations", 2),
        Chunk("Students need 75% attendance for lectures.", "attendance.pdf", "attendance", 1),
    ]
    retriever = TfidfRetriever(chunks)

    results = retriever.search("How late can I enter the examination room?", top_k=2)

    assert isinstance(results[0], RetrievedChunk)
    assert results[0].source == "exam.pdf" and results[0].page == 2
    assert 0 < results[0].score <= 1


# ---- answer evaluation ------------------------------------------------------
class ScriptedAgent:
    """Stands in for AcademicAgent: returns fixed responses, no Gemini needed."""

    def __init__(self, responses):
        self.responses = responses

    def run(self, question):
        return self.responses[question]


def test_evaluate_agent_scores_answers_and_refusals():
    evidence = FakeRetriever().search("attendance", 1)
    responses = {
        QUESTIONS[0]["question"]: RAGResponse(
            QUESTIONS[0]["question"],
            "Students must have at least 80% attendance in a module to sit its final examination [1].",
            evidence),
        QUESTIONS[1]["question"]: RAGResponse(
            QUESTIONS[1]["question"], "I could not find this information.", [], answerable=False),
    }

    result = evaluate_agent(ScriptedAgent(responses), QUESTIONS)
    summary = result["summary"]

    assert summary["groundedness"] == 1.0
    assert summary["answer_relevance"] == 1.0
    assert summary["refusal_accuracy"] == 1.0
    assert summary["false_refusal_rate"] == 0.0


def test_evaluate_agent_skips_gemini_errors():
    failed = RAGResponse("q", "Gemini is busy right now.", [], [AgentStep("observation", "Gemini unavailable: busy")],
                         answerable=False)
    result = evaluate_agent(ScriptedAgent({QUESTIONS[1]["question"]: failed}), [QUESTIONS[1]])

    assert result["summary"]["llm_errors"] == 1
    assert result["summary"]["refusal_accuracy"] is None


# ---- the ground-truth file itself -------------------------------------------
def test_ground_truth_is_well_formed():
    questions = load_ground_truth()

    assert len({q["id"] for q in questions}) == len(questions)
    assert any(not q["answerable"] for q in questions)
    for q in questions:
        if q["answerable"]:
            assert q["expected_sources"] and q["relevant_terms"] and q["expected_keywords"]


@pytest.mark.skipif(not CHUNKS_FILE.exists(), reason="run Member 1's document processor first")
def test_ground_truth_terms_appear_in_their_source_documents():
    """Every expected answer really is in the documents, so a low score is the system's fault."""
    records = [json.loads(line) for line in CHUNKS_FILE.read_text(encoding="utf-8").splitlines() if line]
    for q in load_ground_truth():
        text = " ".join(r["text"] for r in records
                        if r["metadata"]["source"] in q["expected_sources"]).lower()
        for term in q["relevant_terms"] + q["expected_keywords"]:
            assert term.lower() in text, f"{q['id']}: '{term}' not found in {q['expected_sources']}"


@pytest.mark.skipif(not CHUNKS_FILE.exists(), reason="run Member 1's document processor first")
def test_ground_truth_context_is_a_real_passage():
    """Each context passage is copied word for word from one chunk of its source document."""
    norm = lambda s: " ".join(s.split()).lower()
    records = [json.loads(line) for line in CHUNKS_FILE.read_text(encoding="utf-8").splitlines() if line]
    for q in load_ground_truth():
        if q["answerable"]:
            assert any(norm(q["context"]) in norm(r["text"]) for r in records
                       if r["metadata"]["source"] in q["expected_sources"]), q["id"]


# ---- pacing Gemini calls for the free tier ---------------------------------------


class RateLimitError(Exception):
    code = 429


class FlakyLLM:
    """Refuses the first `failures` calls with a 429, then answers."""

    def __init__(self, failures: int):
        self.failures, self.calls = failures, 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        if self.calls <= self.failures:
            raise RateLimitError("429 RESOURCE_EXHAUSTED. Please retry in 40.7s.")
        return "answer"


class FakeClock:
    def __init__(self):
        self.now, self.slept = 0.0, []

    def sleep(self, seconds: float) -> None:
        self.slept.append(round(seconds, 1))
        self.now += seconds


def test_paced_llm_spaces_calls_and_waits_out_a_rate_limit():
    clock = FakeClock()
    llm = PacedLLM(FlakyLLM(failures=1), requests_per_minute=12, sleep=clock.sleep, clock=lambda: clock.now)

    assert llm.generate("q1") == "answer"      # 429, waits 40.7 + 1 s, then answers
    assert llm.generate("q2") == "answer"      # under 5 s since the last call: waits the rest
    assert clock.slept == [41.7, 5.0]
    assert llm.waited_for_limit == 1


def test_paced_llm_gives_up_with_the_agents_error():
    clock = FakeClock()
    llm = PacedLLM(FlakyLLM(failures=10), max_retries=2, give_up_error=RuntimeError,
                   sleep=clock.sleep, clock=lambda: clock.now)

    with pytest.raises(RuntimeError, match="rate limit"):
        llm.generate("q")
    assert retry_delay(Exception("no hint")) == 60.0
