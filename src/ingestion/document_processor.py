
"""Extract, clean, chunk and label academic-administration documents."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

from docx import Document
from pypdf import PdfReader

SUPPORTED = {".pdf", ".docx", ".txt"}

def infer_category(path: Path) -> str:
    """Infer a broad category from the filename; edit mapping for your dataset."""
    name = path.stem.lower().replace("-", " ").replace("_", " ")
    rules = {
        "academic_calendar": ["calendar", "semester", "term dates"],
        "registration": ["registration", "enrolment", "enrollment", "course add"],
        "examinations": ["exam", "assessment", "resit", "repeat", "grading"],
        "attendance": ["attendance", "absence", "leave"],
        "student_services": ["student service", "counselling", "library", "support"],
        "fees_and_payments": ["fee", "payment", "scholarship"],
        "staff": ["lecturer", "staff", "faculty member"],
        "academic_policies": ["policy", "regulation", "handbook", "guideline"],
    }
    for category, words in rules.items():
        if any(word in name for word in words):
            return category
    return "general"

def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[\t\r\f\v]+", " ", text)
    text = re.sub(r" +", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def extract_pages(path: Path) -> list[tuple[int, str]]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return [(i + 1, clean_text(page.extract_text() or "")) for i, page in enumerate(reader.pages)]
    if suffix == ".docx":
        doc = Document(str(path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        # DOCX has no reliable page boundaries in python-docx. Use page_number=None.
        return [(0, clean_text("\n".join(paragraphs)))]
    if suffix == ".txt":
        return [(0, clean_text(path.read_text(encoding="utf-8", errors="replace")))]
    return []

def split_into_chunks(text: str, chunk_size: int = 700, overlap: int = 100) -> list[str]:
    """Split by words, retaining a small overlap to preserve context."""
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(end - overlap, start + 1)
    return chunks

def process_file(path: Path, chunk_size: int = 700, overlap: int = 100) -> list[dict]:
    pages = extract_pages(path)
    records = []
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    for page_num, text in pages:
        for chunk_idx, chunk in enumerate(split_into_chunks(text, chunk_size, overlap), start=1):
            records.append({
                "id": f"{path.stem}-{page_num}-{chunk_idx}",
                "text": chunk,
                "metadata": {
                    "source": path.name,
                    "doc_title": path.stem.replace("_", " ").replace("-", " "),
                    "category": infer_category(path),
                    "file_type": path.suffix.lower().lstrip("."),
                    "page_number": page_num if page_num > 0 else None,
                    "chunk_index": chunk_idx,
                    "word_count": len(chunk.split()),
                    "doc_hash": digest,
                }
            })
    return records

def process_directory(input_dir: Path, output_file: Path) -> int:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    all_records = []
    for path in sorted(input_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            try:
                records = process_file(path)
                all_records.extend(records)
                print(f"[OK] {path.name}: {len(records)} chunks")
            except Exception as exc:
                print(f"[SKIP] {path.name}: {exc}")
    with output_file.open("w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"\nProcessed {len(all_records)} chunks → {output_file}")
    return len(all_records)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/chunks.jsonl"))
    args = parser.parse_args()
    process_directory(args.input, args.output)

if __name__ == "__main__":
    main()
