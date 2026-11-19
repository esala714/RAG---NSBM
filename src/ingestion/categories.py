"""Choosing a document's category from keywords in its file name."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

# The first matching rule wins, so the more specific categories come first.
DEFAULT_RULES: dict[str, list[str]] = {
    "academic_calendar": ["calendar", "semester", "term dates"],
    "registration": ["registration", "enrolment", "enrollment", "course add"],
    "examinations": ["exam", "assessment", "resit", "repeat", "grading"],
    "attendance": ["attendance", "absence", "leave"],
    "student_services": ["student service", "counselling", "library", "support"],
    "fees_and_payments": ["fee", "payment", "scholarship"],
    "staff": ["lecturer", "staff", "faculty member"],
    "academic_policies": ["policy", "regulation", "handbook", "guideline"],
}


class KeywordCategoryClassifier:
    def __init__(self, rules: Mapping[str, Sequence[str]] | None = None, default: str = "general"):
        self.rules = dict(rules or DEFAULT_RULES)
        self.default = default

    def classify(self, path: Path) -> str:
        name = path.stem.lower().replace("-", " ").replace("_", " ")
        for category, words in self.rules.items():
            if any(word in name for word in words):
                return category
        return self.default
