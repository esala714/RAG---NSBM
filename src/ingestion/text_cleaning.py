"""Cleaning of extracted text, one rule per method so each can be tested and explained.

TextCleaner          fixes one page at a time (characters and whitespace)
SplitWordRepairer    fixes words that the PDF text layer split in two ("exa mination"),
                     using the vocabulary of the whole document
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Iterable

Page = tuple[int, str]

_GLYPH_NAME_RE = re.compile(r"/f_(?=[a-z])")      # "/f_irst" -> "first", "in/f_luence" -> "influence"
_WORD_RE = re.compile(r"[A-Za-z]+")
_LEFT_RE = re.compile(r"^([^A-Za-z]*)([A-Za-z]+)$")    # "(exa"      -> "(", "exa"
_RIGHT_RE = re.compile(r"^([A-Za-z]+)([^A-Za-z]*)$")   # "mination," -> "mination", ","


class TextCleaner:
    """Turns raw extracted text into clean text."""

    def clean(self, text: str) -> str:
        text = self.normalise_unicode(text)
        text = self.fix_ligature_glyphs(text)
        return self.normalise_whitespace(text)

    @staticmethod
    def normalise_unicode(text: str) -> str:
        """NFKC turns ligature characters and maths letters into plain ones: "ﬀ" -> "ff", "𝐺𝑃𝐴" -> "GPA"."""
        return unicodedata.normalize("NFKC", text)

    @staticmethod
    def fix_ligature_glyphs(text: str) -> str:
        """Some PDFs write the fi/fl ligature as its glyph name, "/f_i" or "/f_l"."""
        return _GLYPH_NAME_RE.sub("f", text)

    @staticmethod
    def normalise_whitespace(text: str) -> str:
        """Null characters and tabs become spaces, repeated spaces collapse to one,
        and at most one blank line is kept between paragraphs."""
        text = text.replace("\x00", " ")
        text = re.sub(r"[\t\r\f\v]+", " ", text)
        text = re.sub(r" +", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


# Short everyday words are never joined, so "a way" can never become "away".
COMMON_WORDS = frozenset(
    """a an the to in on at of by as is it be or no so up we us he me my do go if
    and for not any all are but can has had was its out our off per new one two""".split()
)


class SplitWordRepairer:
    """Joins two neighbouring words when all of these hold:

    1. the joined word appears intact somewhere in the same document ("examination");
    2. one of the two pieces is rare in the document (at most `max_piece_count` times),
       because real words repeat and broken pieces ("exa", "mination") do not;
    3. neither piece is a common short word, and the joined word has at least `min_length` letters.
    """

    def __init__(self, max_piece_count: int = 2, min_length: int = 4,
                 common_words: Iterable[str] = COMMON_WORDS):
        self.max_piece_count = max_piece_count
        self.min_length = min_length
        self.common_words = frozenset(common_words)

    def repair(self, pages: list[Page]) -> tuple[list[Page], list[str]]:
        """Returns the repaired pages and a log line for every join, e.g. "exa mination -> examination"."""
        counts = Counter(word.lower() for _, text in pages for word in _WORD_RE.findall(text))
        log: list[str] = []
        return [(number, self._repair_text(text, counts, log)) for number, text in pages], log

    def should_join(self, left: str, right: str, counts: Counter) -> bool:
        a, b = left.lower(), right.lower()
        if a in self.common_words or b in self.common_words or len(a + b) < self.min_length:
            return False
        return counts[a + b] >= 1 and min(counts[a], counts[b]) <= self.max_piece_count

    def _repair_text(self, text: str, counts: Counter, log: list[str]) -> str:
        parts = re.split(r"(\s+)", text)          # words at even positions, whitespace at odd ones
        out = [parts[0]]
        i = 1
        while i < len(parts) - 1:
            gap, word = parts[i], parts[i + 1]
            left, right = _LEFT_RE.match(out[-1]), _RIGHT_RE.match(word)
            if gap == " " and left and right and self.should_join(left.group(2), right.group(1), counts):
                merged = out[-1] + word
                log.append(f"{left.group(2)} {right.group(1)} -> {left.group(2)}{right.group(1)}")
                out[-1] = merged                    # the merged word can join the next piece too
            else:
                out += [gap, word]
            i += 2
        return "".join(out)
