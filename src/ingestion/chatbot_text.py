"""Removing what an AI chatbot wrote for its own user from a document.

Eight documents in data/raw were put together from an AI chatbot's answers. Next to the facts,
they keep two things the chatbot wrote for the person it was talking to:

    citation marks    "... at least 50% (combining CA and exam marks). [1, 2]"
                      numbers of web pages that are not in the document
    follow-up offers  "If you'd like, let me know: • Your intake batch ..."

The marks look exactly like the agent's own citations, where [1] means the first retrieved
chunk, so Gemini could copy a mark that points to the wrong chunk. An offer names faculties,
programmes and intakes, so it matches many questions without answering any of them.

ChatbotTextRemover deletes both. An offer is found by how it starts ("If you", "To give you",
...) and by a promise in its first words ("let me know", "I can help"); it is removed from that
line to the end of its paragraph, which holds the list of questions for the user.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

Page = tuple[int, str]

_CITATION_RE = re.compile(r"[ \t]*\[\d+(?:\s*,\s*\d+)*\]")      # " [1]" or " [1, 2, 3]"
_PARAGRAPH_RE = re.compile(r"\n[ ]*\n")
OFFER_STARTS = ("If you", "To give you", "Let me know", "Would you like")
OFFER_PROMISES = ("let me know", "i can help", "i can give you")


@dataclass
class ChatbotRemovals:
    citation_marks: int = 0
    offers: list[str] = field(default_factory=list)     # first words of every removed offer


class ChatbotTextRemover:
    """Deletes citation marks and follow-up offers; `window` is how many words of a line
    starting like an offer are searched for the promise."""

    def __init__(self, window: int = 40):
        if window < 1:
            raise ValueError("window must be at least 1.")
        self.window = window

    def remove(self, pages: list[Page]) -> tuple[list[Page], ChatbotRemovals]:
        removals = ChatbotRemovals()
        return [(number, self._remove_text(text, removals)) for number, text in pages], removals

    def is_offer(self, lines: list[str]) -> bool:
        """True if the first of `lines` starts a follow-up offer."""
        if not lines or not lines[0].strip().startswith(OFFER_STARTS):
            return False
        first_words = " ".join(" ".join(lines).split()[: self.window]).lower()
        return any(promise in first_words for promise in OFFER_PROMISES)

    def _remove_text(self, text: str, removals: ChatbotRemovals) -> str:
        text, marks = _CITATION_RE.subn("", text)
        removals.citation_marks += marks
        kept = []
        for paragraph in _PARAGRAPH_RE.split(text):
            lines = paragraph.split("\n")
            start = next((i for i in range(len(lines)) if self.is_offer(lines[i:])), None)
            if start is not None:
                removals.offers.append(" ".join(" ".join(lines[start:]).split()[:8]) + " ...")
                lines = lines[:start]
            if "\n".join(lines).strip():
                kept.append("\n".join(lines).rstrip())
        return "\n\n".join(kept)
