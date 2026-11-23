"""Extract, clean, chunk and label academic-administration documents.

    extract (BaseExtractor) -> clean (TextCleaner) -> remove chatbot text (ChatbotTextRemover)
    -> repair split words (SplitWordRepairer)
    -> title page as heading (TitlePageHeading) -> chunk (WordChunker) -> label (KeywordCategoryClassifier)

DocumentProcessor receives every part through its constructor (Dependency Injection) and
build_processor() is the one place that chooses the concrete classes. The functions at the
bottom (process_file, SUPPORTED, ...) keep the original interface that Member 2's
scripts/build_index.py and the tests use.

Run:  python -m src.ingestion.document_processor --input data/raw --output data/processed/chunks.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .categories import KeywordCategoryClassifier
from .chatbot_text import ChatbotRemovals, ChatbotTextRemover
from .chunking import CHUNK_OVERLAP, CHUNK_SIZE, WordChunker
from .extractors import BaseExtractor, Page, default_extractors
from .headings import TitlePageHeading
from .text_cleaning import SplitWordRepairer, TextCleaner


@dataclass
class CleaningReport:
    """What was fixed in one document, printed by process_directory()."""

    repairs: list[str] = field(default_factory=list)                # "exa mination -> examination"
    chatbot: ChatbotRemovals = field(default_factory=ChatbotRemovals)

    def summary(self) -> str:
        counts = [(len(self.repairs), "split word", "repaired"),
                  (len(self.chatbot.offers), "chatbot offer", "removed"),
                  (self.chatbot.citation_marks, "citation mark", "removed")]
        parts = [f"{n} {name}{'s' if n > 1 else ''} {done}" for n, name, done in counts if n]
        return f" ({', '.join(parts)})" if parts else ""


class DocumentProcessor:
    def __init__(self, extractors: Sequence[BaseExtractor], cleaner: TextCleaner,
                 chunker: WordChunker, classifier: KeywordCategoryClassifier,
                 repairer: SplitWordRepairer | None = None,
                 heading_finder: TitlePageHeading | None = None,
                 chatbot_remover: ChatbotTextRemover | None = None):
        self.extractors = list(extractors)
        self.cleaner = cleaner
        self.chunker = chunker
        self.classifier = classifier
        self.repairer = repairer      # optional: None keeps the text exactly as cleaned
        self.heading_finder = heading_finder    # optional: None chunks every page as it is
        self.chatbot_remover = chatbot_remover  # optional: None keeps citation marks and offers

    @property
    def supported(self) -> set[str]:
        return {suffix for extractor in self.extractors for suffix in extractor.suffixes}

    def extractor_for(self, path: Path) -> BaseExtractor:
        for extractor in self.extractors:
            if extractor.can_read(path):
                return extractor
        raise ValueError(f"No extractor for '{path.suffix}' files ({path.name}).")

    def read_pages(self, path: Path) -> list[Page]:
        """Extracted, cleaned and repaired text of every page."""
        return self._read_pages(path)[0]

    def _read_pages(self, path: Path) -> tuple[list[Page], CleaningReport]:
        raw_pages = self.extractor_for(path).extract(path)
        pages = [(number, self.cleaner.clean(text)) for number, text in raw_pages]
        report = CleaningReport()
        if self.chatbot_remover is not None:
            pages, report.chatbot = self.chatbot_remover.remove(pages)
        if self.repairer is not None:
            pages, report.repairs = self.repairer.repair(pages)
        return pages, report

    def process_file(self, path: Path) -> list[dict]:
        return self._process(path)[0]

    def _process(self, path: Path) -> tuple[list[dict], CleaningReport]:
        pages, report = self._read_pages(path)
        heading = None
        if self.heading_finder is not None:
            heading, pages = self.heading_finder.split(pages)
        reserve = len(heading.split()) if heading else 0
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        category = self.classifier.classify(path)
        records = []
        for page_num, text in pages:
            for chunk_idx, body in enumerate(self.chunker.split(text, reserve), start=1):
                chunk = f"{heading}: {body}" if heading else body
                records.append({
                    "id": f"{path.stem}-{page_num}-{chunk_idx}",
                    "text": chunk,
                    "metadata": {
                        "source": path.name,
                        "doc_title": path.stem.replace("_", " ").replace("-", " "),
                        "heading": heading,
                        "category": category,
                        "file_type": path.suffix.lower().lstrip("."),
                        "page_number": page_num if page_num > 0 else None,
                        "chunk_index": chunk_idx,
                        "word_count": len(chunk.split()),
                        "doc_hash": digest,
                    },
                })
        return records, report

    def process_directory(self, input_dir: Path, output_file: Path) -> int:
        """Process every supported file under input_dir; a damaged file is skipped, not fatal."""
        output_file.parent.mkdir(parents=True, exist_ok=True)
        all_records = []
        for path in sorted(input_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in self.supported:
                try:
                    records, report = self._process(path)
                    all_records.extend(records)
                    print(f"[OK] {path.name}: {len(records)} chunks{report.summary()}")
                except Exception as exc:
                    print(f"[SKIP] {path.name}: {exc}")
        with output_file.open("w", encoding="utf-8") as f:
            for record in all_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"\nProcessed {len(all_records)} chunks → {output_file}")
        return len(all_records)


def build_processor(chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> DocumentProcessor:
    """Composition root: the only place that decides which concrete classes are used."""
    return DocumentProcessor(
        extractors=default_extractors(),
        cleaner=TextCleaner(),
        chunker=WordChunker(chunk_size, overlap),
        classifier=KeywordCategoryClassifier(),
        repairer=SplitWordRepairer(),
        heading_finder=TitlePageHeading(),
        chatbot_remover=ChatbotTextRemover(),
    )


# --------------------------------------------------------------------------- #
# Function interface (used by Member 2's scripts/build_index.py and the tests) #
# --------------------------------------------------------------------------- #
SUPPORTED = build_processor().supported


def infer_category(path: Path) -> str:
    return KeywordCategoryClassifier().classify(path)


def clean_text(text: str) -> str:
    return TextCleaner().clean(text)


def extract_pages(path: Path) -> list[Page]:
    return build_processor().read_pages(path)


def split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    return WordChunker(chunk_size, overlap).split(text)


def process_file(path: Path, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    return build_processor(chunk_size, overlap).process_file(path)


def process_directory(input_dir: Path, output_file: Path,
                      chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> int:
    return build_processor(chunk_size, overlap).process_directory(input_dir, output_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="words per chunk")
    parser.add_argument("--overlap", type=int, default=CHUNK_OVERLAP, help="words shared by neighbouring chunks")
    args = parser.parse_args()
    process_directory(args.input, args.output, args.chunk_size, args.overlap)


if __name__ == "__main__":
    main()
