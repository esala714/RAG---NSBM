import time

from google import genai
from google.genai import errors

from src.core.interfaces import BaseLLMClient


class LLMUnavailableError(Exception):
    """Raised when Gemini is still busy after several tries."""


class GeminiClient(BaseLLMClient):
    def __init__(self, api_key: str, model_name: str, temperature: float = 0.0,
                 max_attempts: int = 3, wait_seconds: int = 5):
        self._client = genai.Client(api_key=api_key)
        self._model = model_name
        self._temperature = temperature   # 0 = less creative = fewer hallucinations
        self._max_attempts = max_attempts
        self._wait_seconds = wait_seconds

    def generate(self, prompt: str) -> str:
        for attempt in range(1, self._max_attempts + 1):
            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=prompt,
                    config={"temperature": self._temperature},
                )
                return response.text or ""
            except errors.ServerError as error:   # 5xx errors, e.g. 503 busy
                if attempt == self._max_attempts:
                    raise LLMUnavailableError(
                        "Gemini is busy right now. Please try again in a few minutes."
                    ) from error
                time.sleep(self._wait_seconds * attempt)   # wait 5s, then 10s
        return ""




    