"""Text extractors: one class per file type, all behind the BaseExtractor interface.

This is the Strategy pattern. DocumentProcessor only knows "an extractor gives me
(page number, text) pairs", so a new format (e.g. HTML, or OCR for scanned PDFs) is one
new class, with no change to the cleaning, chunking or labelling code.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from docx import Document
from pypdf import PdfReader

Page = tuple[int, str]   # (page number, raw text); page 0 = the format has no page numbers


class BaseExtractor(ABC):
    """Rule: every extractor says which file types it reads and returns its pages."""

    suffixes: tuple[str, ...] = ()

    def can_read(self, path: Path) -> bool:
        return path.suffix.lower() in self.suffixes

    @abstractmethod
    def extract(self, path: Path) -> list[Page]:
        """Return (page number, raw text) for every page of the file."""


class PdfExtractor(BaseExtractor):
    """PDF files, page by page with pypdf. Page numbers start at 1."""

    suffixes = (".pdf",)

    def extract(self, path: Path) -> list[Page]:
        reader = PdfReader(str(path))
        return [(number, page.extract_text() or "") for number, page in enumerate(reader.pages, start=1)]


class DocxExtractor(BaseExtractor):
    """Word files with python-docx: paragraphs first, then every table row as "cell | cell".
    DOCX has no reliable page boundaries, so the whole file is page 0."""

    suffixes = (".docx",)

    def extract(self, path: Path) -> list[Page]:
        doc = Document(str(path))
        lines = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if cells:
                    lines.append(" | ".join(cells))
        return [(0, "\n".join(lines))]


class TxtExtractor(BaseExtractor):
    """Plain UTF-8 text files (page 0)."""

    suffixes = (".txt",)

    def extract(self, path: Path) -> list[Page]:
        return [(0, path.read_text(encoding="utf-8", errors="replace"))]


def default_extractors() -> list[BaseExtractor]:
    return [PdfExtractor(), DocxExtractor(), TxtExtractor()]
