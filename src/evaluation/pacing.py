"""Keeping a long evaluation run under the Gemini free-tier rate limit.

The free tier allows 15 generate requests per minute per model, and the agent makes several per
question (search plan, query rewrites, answer, answer check). Without pacing, an evaluation of
17 questions stops after a few of them with "429 RESOURCE_EXHAUSTED".

PacedLLM wraps any LLM client and
  1. leaves at least 60 / requests_per_minute seconds between two calls, and
  2. on a 429 waits as long as the error asks ("Please retry in 40.7s") and tries again.
If Gemini still refuses after `max_retries` waits, it raises the agent's LLMUnavailableError, so
the agent reports "Gemini unavailable" and the evaluation counts an LLM error instead of crashing.
"""
from __future__ import annotations

import re
import time
from typing import Callable

from src.core.interfaces import BaseLLMClient

_RETRY_IN_RE = re.compile(r"retry in ([\d.]+)\s*s", re.IGNORECASE)
DEFAULT_RETRY_SECONDS = 60.0


def is_rate_limit(error: Exception) -> bool:
    return getattr(error, "code", None) == 429 or "RESOURCE_EXHAUSTED" in str(error)


def retry_delay(error: Exception) -> float:
    """Seconds the 429 error asks us to wait (60 if it does not say)."""
    match = _RETRY_IN_RE.search(str(error))
    return float(match.group(1)) if match else DEFAULT_RETRY_SECONDS


class PacedLLM(BaseLLMClient):
    def __init__(self, llm: BaseLLMClient, requests_per_minute: float = 12, max_retries: int = 5,
                 give_up_error: type[Exception] | None = None,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic):
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be greater than 0.")
        self.llm = llm
        self.interval = 60.0 / requests_per_minute
        self.max_retries = max_retries
        self._give_up_error = give_up_error
        self._sleep = sleep
        self._clock = clock
        self._last_call: float | None = None
        self.waited_for_limit = 0       # how many 429 errors were waited out

    def generate(self, prompt: str) -> str:
        for attempt in range(self.max_retries + 1):
            self._wait_turn()
            try:
                return self.llm.generate(prompt)
            except Exception as error:
                if not is_rate_limit(error):
                    raise
                if attempt == self.max_retries:
                    raise self._unavailable("Gemini rate limit: still exhausted after "
                                            f"{self.max_retries} waits.") from error
                self.waited_for_limit += 1
                self._sleep(retry_delay(error) + 1)
        return ""

    def _wait_turn(self) -> None:
        now = self._clock()
        if self._last_call is not None:
            remaining = self._last_call + self.interval - now
            if remaining > 0:
                self._sleep(remaining)
        self._last_call = self._clock()

    def _unavailable(self, message: str) -> Exception:
        error_class = self._give_up_error
        if error_class is None:     # imported here: the Gemini package is only needed with --with-agent
            from src.rag.llm_client import LLMUnavailableError as error_class
        return error_class(message)
