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
from src.ingestion.headings import TitlePageHeading
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


# ---- PDF text repairs ---------------------------------------------------------
from src.ingestion.text_cleaning import SplitWordRepairer  # noqa: E402


def test_cleaner_fixes_ligatures_and_maths_letters():
    cleaner = TextCleaner()

    assert cleaner.clean("eﬀective and eﬃcient") == "effective and efficient"   # ﬀ, ﬃ
    assert cleaner.clean("the /f_irst year and in/f_luence") == "the first year and influence"
    assert cleaner.clean("\U0001d43a\U0001d443\U0001d434 = 3.7") == "GPA = 3.7"          # 𝐺𝑃𝐴


def test_repairer_joins_a_split_word_seen_intact_in_the_document():
    pages = [(1, "Read the examination rules."), (2, "The exa mination starts at 9.")]

    repaired, log = SplitWordRepairer().repair(pages)

    assert repaired[1] == (2, "The examination starts at 9.")
    assert log == ["exa mination -> examination"]


def test_repairer_never_joins_common_words_or_unknown_words():
    pages = [(1, "Go away now. There is a way out."),           # "a way" must not become "away"
             (2, "The rule book lists every rule.")]           # "rulebook" never appears intact

    repaired, log = SplitWordRepairer().repair(pages)

    assert repaired == pages
    assert log == []


# ---- title pages ----------------------------------------------------------------


def test_title_pages_need_a_longer_page_after_them():
    finder = TitlePageHeading(max_title_words=5)
    short_only = [(1, "Students must attend"), (2, "lectures")]

    assert finder.split(short_only) == (None, short_only)
    assert finder.split([(1, "one two three four five six")]) == (None, [(1, "one two three four five six")])


def test_a_long_title_page_is_dropped_but_is_not_a_heading():
    finder = TitlePageHeading(max_title_words=10, max_heading_words=3)
    content = (2, "one two three four five six seven eight nine ten eleven")

    assert finder.split([(1, "MBA HANDBOOK MINISTRY APPROVED"), content]) == (None, [content])
    assert finder.split([(1, "MBA HANDBOOK"), content]) == ("MBA HANDBOOK", [content])


def test_chunker_keeps_room_for_a_heading():
    chunks = WordChunker(chunk_size=4, overlap=1).split("a b c d e f", reserve=1)

    assert chunks == ["a b c", "c d e", "e f"]
