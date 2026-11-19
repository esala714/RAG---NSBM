"""Tests for the ingestion classes.   Run:  python -m pytest -q tests/test_ingestion_classes.py

They need no NSBM documents: every file is created in a temporary folder.
"""
from pathlib import Path

import pytest
from docx import Document

from src.ingestion.categories import KeywordCategoryClassifier
from src.ingestion.chunking import WordChunker
from src.ingestion.document_processor import DocumentProcessor, build_processor
from src.ingestion.extractors import BaseExtractor, DocxExtractor
from src.ingestion.text_cleaning import TextCleaner


def test_docx_extraction_reads_paragraphs_and_tables(tmp_path):
    path = tmp_path / "NSBM Leave Policy.docx"
    doc = Document()
    doc.add_paragraph("Students must apply for leave in advance.")
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Medical leave"
    table.rows[0].cells[1].text = "Certificate required"
    doc.save(path)

    pages = DocxExtractor().extract(path)
    records = build_processor().process_file(path)

    assert pages[0][0] == 0                                   # DOCX has no page numbers
    assert "Medical leave | Certificate required" in pages[0][1]
    assert records[0]["text"] == ("Students must apply for leave in advance. "
                                  "Medical leave | Certificate required")
    assert records[0]["metadata"]["file_type"] == "docx"
    assert records[0]["metadata"]["page_number"] is None
    assert records[0]["metadata"]["category"] == "attendance"


class FakeExtractor(BaseExtractor):
    """Stands in for a real extractor: shows that DocumentProcessor works with any BaseExtractor."""

    suffixes = (".fake",)

    def extract(self, path):
        return [(1, "one two three four five"), (2, "six seven")]


def test_processor_uses_the_injected_parts(tmp_path):
    path = tmp_path / "exam rules.fake"
    path.write_text("ignored")
    processor = DocumentProcessor(
        extractors=[FakeExtractor()],
        cleaner=TextCleaner(),
        chunker=WordChunker(chunk_size=3, overlap=1),
        classifier=KeywordCategoryClassifier({"examinations": ["exam"]}),
    )

    records = processor.process_file(path)

    assert [r["text"] for r in records] == ["one two three", "three four five", "six seven"]
    assert [r["metadata"]["page_number"] for r in records] == [1, 1, 2]
    assert records[0]["metadata"]["category"] == "examinations"
    assert processor.supported == {".fake"}


def test_unknown_file_type_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        build_processor().extractor_for(tmp_path / "notes.xlsx")


def test_chunker_rejects_bad_settings():
    with pytest.raises(ValueError):
        WordChunker(chunk_size=0)
    with pytest.raises(ValueError):
        WordChunker(overlap=-1)


def test_damaged_file_is_skipped_not_fatal(tmp_path, capsys):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "broken.pdf").write_bytes(b"this is not a real pdf")
    (raw / "NSBM Attendance Policy.txt").write_text("Students must attend lectures.", encoding="utf-8")

    count = build_processor().process_directory(raw, tmp_path / "chunks.jsonl")

    assert count == 1
    assert "[SKIP] broken.pdf" in capsys.readouterr().out
