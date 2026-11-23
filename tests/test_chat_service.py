"""Tests for Backend/chat_service.py (the UI chat -> AcademicAgent connection).

Run from the project folder:  python -m pytest tests/test_chat_service.py
No Gemini key or vector database is needed: the agent gets a fake retriever and a fake LLM.
"""

import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Backend"))

pytest.importorskip("google.genai")   # the agent imports the Gemini client

import chat_service  # noqa: E402
from chat_service import ChatError  # noqa: E402
from src.agent.academic_agent import AcademicAgent  # noqa: E402
from src.core.interfaces import BaseLLMClient  # noqa: E402
from src.rag.prompt_builder import PromptBuilder  # noqa: E402
from tests.fake_retriever import EmptyRetriever, FakeRetriever  # noqa: E402


class FakeLLM(BaseLLMClient):
    """Stands in for Gemini: plans one query and answers with a citation."""

    def __init__(self, answer="You need at least **80%** attendance to sit the exam [1]."):
        self.answer = answer

    def generate(self, prompt: str) -> str:
        if "JSON list" in prompt:
            return json.dumps(["minimum attendance requirement"])
        if "Rewrite it" in prompt:
            return "minimum attendance requirement for examinations"
        return self.answer


class BrokenAgent:
    def __init__(self, error):
        self.error = error

    def run(self, question):
        raise self.error


@pytest.fixture
def use_agent(tmp_path):
    def _use(retriever=None, llm=None):
        agent = AcademicAgent(retriever or FakeRetriever(), llm or FakeLLM(), PromptBuilder(),
                              trace_dir=str(tmp_path / "traces"))
        chat_service.set_agent(agent)
        return agent
    yield _use
    chat_service.set_agent(None)


def test_answer_has_the_answer_sources_and_trace(use_agent):
    use_agent()
    result = chat_service.answer("  What attendance do I need   to sit an exam? ")

    assert result["question"] == "What attendance do I need to sit an exam?"
    assert result["answerable"] is True
    assert "80%" in result["answer"] and "[1]" in result["answer"]
    first = result["sources"][0]
    assert first["number"] == 1
    assert first["source"] == "exam_regulations_SAMPLE.pdf" and first["page"] == 4
    assert first["excerpt"].startswith("SAMPLE: Students must have at least 80% attendance")
    phases = [step["phase"] for step in result["trace"]]
    assert phases[0] == "observation" and phases[-1] == "persistence"
    json.dumps(result)   # the API can send it as JSON


def test_numbers_match_the_citations_in_the_answer(use_agent):
    use_agent()
    result = chat_service.answer("Can I repeat a missed exam?")
    assert [s["number"] for s in result["sources"]] == list(range(1, len(result["sources"]) + 1))


def test_not_found_has_no_sources(use_agent):
    use_agent(retriever=EmptyRetriever())
    result = chat_service.answer("What is on the canteen menu?")
    assert result["answerable"] is False
    assert result["answer"] == PromptBuilder.NOT_FOUND
    assert result["sources"] == []


def test_long_excerpts_are_shortened(use_agent):
    use_agent()
    chunk = FakeRetriever().search("x", 1)[0]
    chunk.text = "word " * 200
    assert len(chat_service._chunk_json(1, chunk)["excerpt"]) <= chat_service.EXCERPT_LENGTH + 3


def test_trace_shows_paths_from_the_project_folder(use_agent):
    agent = use_agent()
    agent.trace_dir = ROOT / "logs" / "test_traces"
    try:
        result = chat_service.answer("What attendance do I need?")
        saved = result["trace"][-1]["detail"]
        assert str(ROOT) not in saved and "logs" in saved
    finally:
        for file in agent.trace_dir.glob("trace_*.json"):
            file.unlink()
        agent.trace_dir.rmdir()


@pytest.mark.parametrize("question", ["", "   ", None])
def test_empty_question_is_refused(use_agent, question):
    use_agent()
    with pytest.raises(ChatError) as error:
        chat_service.answer(question)
    assert error.value.status == 422 and error.value.code == "empty_question"


def test_very_long_question_is_refused(use_agent):
    use_agent()
    with pytest.raises(ChatError) as error:
        chat_service.answer("why " * 400)
    assert error.value.code == "question_too_long"


def test_missing_gemini_key_explains_the_fix(monkeypatch):
    chat_service.set_agent(None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ChatError) as error:
        chat_service.answer("hello")
    assert error.value.status == 503 and "GEMINI_API_KEY" in error.value.message


def test_missing_index_explains_the_fix(monkeypatch, tmp_path):
    chat_service.set_agent(None)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "no_index_here"))
    with pytest.raises(ChatError) as error:
        chat_service.answer("hello")
    assert error.value.status == 503 and "build_index.py" in error.value.message


def test_wrong_api_key_gets_a_clear_message():
    chat_service.set_agent(BrokenAgent(RuntimeError("400 INVALID_ARGUMENT. API key not valid.")))
    try:
        with pytest.raises(ChatError) as error:
            chat_service.answer("hello")
        assert error.value.status == 502 and "GEMINI_API_KEY" in error.value.message
    finally:
        chat_service.set_agent(None)


def test_other_errors_point_to_the_terminal():
    chat_service.set_agent(BrokenAgent(ValueError("something odd")))
    try:
        with pytest.raises(ChatError) as error:
            chat_service.answer("hello")
        assert error.value.status == 500 and "terminal" in error.value.message
    finally:
        chat_service.set_agent(None)


def test_agent_is_built_like_run_demo_with_short_gemini_waits(monkeypatch, tmp_path):
    """The chat builds the same agent as run_demo.py, but gives up sooner on Gemini's limits."""
    import src.rag.llm_client as llm_client

    made = {}

    class SpyGemini(FakeLLM):
        def __init__(self, api_key, model_name, **options):
            super().__init__()
            made.update(api_key=api_key, model=model_name, **options)

    retriever = FakeRetriever()
    retriever.store = types.SimpleNamespace(count=lambda: 148)
    fake_retrieval = types.ModuleType("src.retrieval")
    fake_retrieval.build_retriever = lambda: retriever
    monkeypatch.setitem(sys.modules, "src.retrieval", fake_retrieval)
    monkeypatch.setattr(llm_client, "GeminiClient", SpyGemini)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    (tmp_path / "vector_db").mkdir()
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "vector_db"))
    chat_service.set_agent(None)
    try:
        agent = chat_service.get_agent()
        assert agent.retriever is retriever
        assert made == {"api_key": "test-key", "model": "test-model",
                        "rate_limit_waits": chat_service.RATE_LIMIT_WAITS,
                        "max_rate_limit_wait": chat_service.MAX_RATE_LIMIT_WAIT}
        assert chat_service.MAX_RATE_LIMIT_WAIT <= 30
        assert agent.trace_dir == ROOT / "logs" / "traces"
    finally:
        chat_service.set_agent(None)


def test_empty_index_explains_the_fix(monkeypatch, tmp_path):
    retriever = FakeRetriever()
    retriever.store = types.SimpleNamespace(count=lambda: 0)
    fake_retrieval = types.ModuleType("src.retrieval")
    fake_retrieval.build_retriever = lambda: retriever
    monkeypatch.setitem(sys.modules, "src.retrieval", fake_retrieval)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path))
    chat_service.set_agent(None)
    with pytest.raises(ChatError) as error:
        chat_service.answer("hello")
    assert error.value.status == 503 and "empty" in error.value.message


def test_status_reports_what_is_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path))
    status = chat_service.status()
    assert status["gemini_key_set"] is False and status["vector_db_found"] is True
