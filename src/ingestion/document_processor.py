"""Extract, clean, chunk and label academic-administration documents.

    extract (BaseExtractor) -> clean (TextCleaner) -> chunk (WordChunker) -> label (KeywordCategoryClassifier)

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
from pathlib import Path
from typing import Sequence

from .categories import KeywordCategoryClassifier
from .chunking import CHUNK_OVERLAP, CHUNK_SIZE, WordChunker
from .extractors import BaseExtractor, Page, default_extractors
from .text_cleaning import TextCleaner


class DocumentProcessor:
    def __init__(self, extractors: Sequence[BaseExtractor], cleaner: TextCleaner,
                 chunker: WordChunker, classifier: KeywordCategoryClassifier):
        self.extractors = list(extractors)
        self.cleaner = cleaner
        self.chunker = chunker
        self.classifier = classifier

    @property
    def supported(self) -> set[str]:
        return {suffix for extractor in self.extractors for suffix in extractor.suffixes}

    def extractor_for(self, path: Path) -> BaseExtractor:
        for extractor in self.extractors:
            if extractor.can_read(path):
                return extractor
        raise ValueError(f"No extractor for '{path.suffix}' files ({path.name}).")

    def read_pages(self, path: Path) -> list[Page]:
        """Extracted and cleaned text of every page."""
        raw_pages = self.extractor_for(path).extract(path)
        return [(number, self.cleaner.clean(text)) for number, text in raw_pages]

    def process_file(self, path: Path) -> list[dict]:
        pages = self.read_pages(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        category = self.classifier.classify(path)
        records = []
        for page_num, text in pages:
            for chunk_idx, chunk in enumerate(self.chunker.split(text), start=1):
                records.append({
                    "id": f"{path.stem}-{page_num}-{chunk_idx}",
                    "text": chunk,
                    "metadata": {
                        "source": path.name,
                        "doc_title": path.stem.replace("_", " ").replace("-", " "),
                        "category": category,
                        "file_type": path.suffix.lower().lstrip("."),
                        "page_number": page_num if page_num > 0 else None,
                        "chunk_index": chunk_idx,
                        "word_count": len(chunk.split()),
                        "doc_hash": digest,
                    },
                })
        return records

    def process_directory(self, input_dir: Path, output_file: Path) -> int:
        """Process every supported file under input_dir; a damaged file is skipped, not fatal."""
        output_file.parent.mkdir(parents=True, exist_ok=True)
        all_records = []
        for path in sorted(input_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in self.supported:
                try:
                    records = self.process_file(path)
                    all_records.extend(records)
                    print(f"[OK] {path.name}: {len(records)} chunks")
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
