"""
Connects the chat box in Frontend/UI.html to the team's AcademicAgent (src/agent).

auth_api.py calls answer(question) from POST /api/chat. The agent is built once,
the same way run_demo.py builds it (hybrid retriever + Gemini + PromptBuilder),
and then reused for every question.

What the agent needs (otherwise the chat explains what is missing):
    * Python 3.11 or newer (the team code uses 3.10+ syntax)
    * pip install -r requirements.txt
    * GEMINI_API_KEY (and GEMINI_MODEL) in the project's .env file
    * the document index: python scripts/build_index.py --raw data/raw --reset
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

# The project folder (it holds src/, data/ and vector_db/), so "import src..." works
# whichever folder uvicorn is started from.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MAX_QUESTION_LENGTH = 1000
EXCERPT_LENGTH = 300
DEFAULT_MODEL = "gemini-3.5-flash-lite"


class ChatError(Exception):
    """A problem the user should see. auth_api.py turns it into a JSON error."""

    def __init__(self, code: str, message: str, status: int):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


def unavailable(message: str) -> ChatError:
    """The agent cannot run yet; the message says how to fix it."""
    return ChatError("chat_unavailable", message, 503)


_agent: Any = None
_build_lock = threading.Lock()   # build the agent only once
_run_lock = threading.Lock()     # one question at a time (Gemini free tier has low limits)


def _chroma_dir() -> Path:
    path = Path(os.getenv("CHROMA_DIR") or "vector_db")
    return path if path.is_absolute() else ROOT / path


def status() -> dict:
    """Shown by GET /api/health so you can see what the chat still needs."""
    return {
        "agent_ready": _agent is not None,
        "python_ok": sys.version_info >= (3, 10),
        "gemini_key_set": bool(os.getenv("GEMINI_API_KEY")),
        "vector_db_found": _chroma_dir().is_dir(),
    }


def _build_agent() -> Any:
    if sys.version_info < (3, 10):
        raise unavailable(
            "The AI agent needs Python 3.11 or newer, but the backend is running on "
            f"Python {sys.version_info.major}.{sys.version_info.minor}. "
            "Start uvicorn from the Python 3.11+ venv."
        )
    if not os.getenv("GEMINI_API_KEY"):
        raise unavailable(
            "GEMINI_API_KEY is not set. Add it to the .env file in the project folder, "
            "then restart the backend."
        )
    chroma = _chroma_dir()
    if not chroma.is_dir():
        raise unavailable(
            "The document index has not been built yet. In the project folder run: "
            "python scripts/build_index.py --raw data/raw --reset"
        )
    os.environ["CHROMA_DIR"] = str(chroma)   # absolute, so the start folder doesn't matter

    try:
        from src.agent.academic_agent import AcademicAgent
        from src.rag.llm_client import GeminiClient
        from src.rag.prompt_builder import PromptBuilder
        from src.retrieval import build_retriever
    except ImportError as error:
        raise unavailable(
            f"A package the agent needs is missing ({getattr(error, 'name', None) or error}). "
            "Run: pip install -r requirements.txt"
        ) from error

    retriever = build_retriever()
    if retriever.store.count() == 0:
        raise unavailable(
            "The document index is empty. In the project folder run: "
            "python scripts/build_index.py --raw data/raw --reset"
        )
    return AcademicAgent(
        retriever=retriever,
        llm=GeminiClient(os.getenv("GEMINI_API_KEY"), os.getenv("GEMINI_MODEL") or DEFAULT_MODEL),
        prompt_builder=PromptBuilder(),
        trace_dir=str(ROOT / "logs" / "traces"),
    )


def get_agent() -> Any:
    global _agent
    if _agent is None:
        with _build_lock:
            if _agent is None:
                _agent = _build_agent()
    return _agent


def set_agent(agent: Any) -> None:
    """Tests use this to plug in an agent with a fake retriever and fake LLM."""
    global _agent
    _agent = agent


def warm_up() -> None:
    """Build the agent and load the search model at startup, so the first question is fast."""
    try:
        agent = get_agent()
        with _run_lock:
            agent.retriever.search("warm up", 1)   # downloads MiniLM the first time, builds TF-IDF
        print("[chat] AI agent ready.", flush=True)
    except ChatError as error:
        print(f"[chat] Chat is not ready yet: {error.message}", flush=True)
    except Exception as error:   # noqa: BLE001 - only a startup message
        print(f"[chat] Chat is not ready yet: {error}", flush=True)


def _friendly(error: Exception) -> ChatError:
    """Turn an unexpected error (usually from Gemini) into a message the user can act on."""
    text = str(error)
    lower = f"{type(error).__name__} {text}".lower()
    print(f"[chat] Agent error: {type(error).__name__}: {text}", flush=True)
    if "api key" in lower or "api_key" in lower:
        return ChatError("chat_failed",
                         "Gemini rejected the API key. Check GEMINI_API_KEY in .env, "
                         "then restart the backend.", 502)
    if "not found" in lower and "model" in lower:
        model = os.getenv("GEMINI_MODEL") or DEFAULT_MODEL
        return ChatError("chat_failed",
                         f"Gemini doesn't know the model '{model}'. Check GEMINI_MODEL in .env.", 502)
    if any(word in lower for word in ("connect", "timed out", "timeout", "name resolution", "servname")):
        return ChatError("chat_failed",
                         "The backend can't reach Gemini. Check the internet connection.", 502)
    return ChatError("chat_failed", "The AI agent hit an error. See the backend terminal for details.", 500)


def _chunk_json(number: int, chunk: Any) -> dict:
    text = " ".join(str(chunk.text).split())
    if len(text) > EXCERPT_LENGTH:
        text = text[:EXCERPT_LENGTH].rsplit(" ", 1)[0] + "..."
    extra = getattr(chunk, "extra", None) or {}
    return {
        "number": number,   # matches the [n] the answer cites
        "source": chunk.source,
        "page": chunk.page,
        "score": round(float(chunk.score), 3),
        "category": extra.get("category"),
        "excerpt": text,
    }


def answer(question: Optional[str]) -> dict:
    """Ask the agent one question and return the answer, its sources and its trace."""
    question = " ".join((question or "").split())
    if not question:
        raise ChatError("empty_question", "Please type a question.", 422)
    if len(question) > MAX_QUESTION_LENGTH:
        raise ChatError("question_too_long",
                        f"Please keep questions under {MAX_QUESTION_LENGTH} characters.", 422)

    agent = get_agent()
    started = time.monotonic()
    try:
        with _run_lock:
            result = agent.run(question)
    except ChatError:
        raise
    except Exception as error:   # noqa: BLE001
        raise _friendly(error) from error

    return {
        "question": result.question,
        "answer": result.answer,
        "answerable": bool(result.answerable),
        "sources": [_chunk_json(i, c) for i, c in enumerate(result.sources, start=1)],
        # paths are shown from the project folder ("logs/traces/...") rather than the full disk path
        "trace": [{"phase": step.phase, "detail": str(step.detail).replace(str(ROOT) + os.sep, "")}
                  for step in result.trace],
        "seconds": round(time.monotonic() - started, 1),
    }
