import re
import time

from google import genai
from google.genai import errors

from src.core.interfaces import BaseLLMClient

# Gemini's 429 message often says how long to wait, e.g. "Please retry in 40.7s."
_RETRY_IN = re.compile(r"retry in ([\d.]+)\s*s", re.IGNORECASE)


class LLMUnavailableError(Exception):
    """Raised when Gemini cannot answer right now (busy, or the rate limit is reached)."""


class RateLimitError(LLMUnavailableError):
    """Raised when Gemini still says 429 (too many requests) after waiting.

    code = 429 lets the evaluation's PacedLLM recognise it and wait longer.
    """
    code = 429


class GeminiClient(BaseLLMClient):
    def __init__(self, api_key: str, model_name: str, temperature: float = 0.0,
                 max_attempts: int = 3, wait_seconds: int = 5,
                 rate_limit_waits: int = 2, max_rate_limit_wait: float = 60):
        self._client = genai.Client(api_key=api_key)
        self._model = model_name
        self._temperature = temperature   # 0 = less creative = fewer hallucinations
        self._max_attempts = max_attempts
        self._wait_seconds = wait_seconds
        self._rate_limit_waits = rate_limit_waits          # how often to wait out a 429
        self._max_rate_limit_wait = max_rate_limit_wait    # never wait longer than this

    def generate(self, prompt: str) -> str:
        busy_tries = 0
        limit_waits = 0
        while True:
            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=prompt,
                    config={"temperature": self._temperature},
                )
                return response.text or ""
            except errors.ServerError as error:   # 5xx errors, e.g. 503 busy
                busy_tries += 1
                if busy_tries >= self._max_attempts:
                    raise LLMUnavailableError(
                        "Gemini is busy right now. Please try again in a few minutes."
                    ) from error
                time.sleep(self._wait_seconds * busy_tries)   # wait 5s, then 10s
            except errors.ClientError as error:   # 4xx errors
                if error.code != 429:              # e.g. a wrong API key: show it as it is
                    raise
                if limit_waits >= self._rate_limit_waits:
                    raise RateLimitError(
                        "Gemini's free-tier request limit was reached. "
                        "Please wait a minute and try again."
                    ) from error
                limit_waits += 1
                time.sleep(self._rate_limit_delay(error))

    def _rate_limit_delay(self, error: Exception) -> float:
        """Seconds to wait after a 429: what Gemini asks for (+1s), or 20s if it doesn't say."""
        match = _RETRY_IN.search(str(error))
        seconds = float(match.group(1)) + 1 if match else 20
        return min(seconds, self._max_rate_limit_wait)