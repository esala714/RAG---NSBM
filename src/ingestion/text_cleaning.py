"""Cleaning of extracted text, one rule per method so each can be tested and explained."""
from __future__ import annotations

import re


class TextCleaner:
    """Turns raw extracted text into clean text."""

    def clean(self, text: str) -> str:
        return self.normalise_whitespace(text)

    @staticmethod
    def normalise_whitespace(text: str) -> str:
        """Null characters and tabs become spaces, repeated spaces collapse to one,
        and at most one blank line is kept between paragraphs."""
        text = text.replace("\x00", " ")
        text = re.sub(r"[\t\r\f\v]+", " ", text)
        text = re.sub(r" +", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
