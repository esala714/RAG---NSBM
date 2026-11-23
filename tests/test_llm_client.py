from types import SimpleNamespace

import pytest
from google.genai import errors

from src.evaluation.pacing import is_rate_limit
from src.rag.llm_client import GeminiClient, LLMUnavailableError, RateLimitError


def rate_limit_error():
    return errors.ClientError(429, {"error": {
        "code": 429, "status": "RESOURCE_EXHAUSTED",
        "message": "Quota exceeded. Please retry in 1.5s."}})


class FakeModels:
    """Pretends to be Gemini: returns or raises the given outcomes in order."""
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def generate_content(self, **kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(text=outcome)


def make_client(outcomes, monkeypatch, sleeps):
    monkeypatch.setattr("src.rag.llm_client.time.sleep", sleeps.append)   # no real waiting
    client = GeminiClient(api_key="test-key", model_name="test-model")
    client._client = SimpleNamespace(models=FakeModels(outcomes))
    return client


def test_waits_out_one_rate_limit_then_answers(monkeypatch):
    sleeps = []
    client = make_client([rate_limit_error(), "Hello"], monkeypatch, sleeps)
    assert client.generate("hi") == "Hello"
    assert sleeps == [2.5]          # Gemini asked for 1.5s, plus 1s


def test_gives_up_with_clear_error_that_pacing_recognises(monkeypatch):
    sleeps = []
    client = make_client([rate_limit_error()] * 3, monkeypatch, sleeps)
    with pytest.raises(RateLimitError) as caught:
        client.generate("hi")
    assert isinstance(caught.value, LLMUnavailableError)   # the agent catches it, no crash
    assert is_rate_limit(caught.value)                     # Member 1's PacedLLM still sees a 429
    assert client._client.models.calls == 3                # 1 try + 2 waits


def test_other_client_errors_are_not_retried(monkeypatch):
    sleeps = []
    bad_key = errors.ClientError(400, {"error": {
        "code": 400, "status": "INVALID_ARGUMENT", "message": "API key not valid."}})
    client = make_client([bad_key], monkeypatch, sleeps)
    with pytest.raises(errors.ClientError):
        client.generate("hi")
    assert sleeps == []