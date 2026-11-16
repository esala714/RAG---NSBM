"""Build (or update) the persistent Chroma index from Member 1's chunks.

Normal use - Member 1 exports chunks as JSON Lines, one chunk per line:
    {"text": "...", "source": "NSBM_Leave_Policy.pdf", "category": "leave_policy", "page": 2, "chunk_index": 0}

    python scripts/build_index.py --chunks data/processed/chunks.jsonl

Before Member 1's module is merged you can test with the raw folder
(a SIMPLE stand-in loader, not the real ingestion step):

    python scripts/build_index.py --raw data/raw --reset
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retrieval import Chunk, HybridRetriever, build_store  # noqa: E402
from src.retrieval.vector_math import magnitude  # noqa: E402


def load_jsonl(path: Path) -> list[Chunk]:
    chunks = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if line.strip():
                try:
                    chunks.append(Chunk.from_any(json.loads(line)))
                except (ValueError, json.JSONDecodeError) as exc:
                    print(f"  skipped line {line_no}: {exc}")
    return chunks


# --------------------------------------------------------------------------- #
# Temporary stand-in for Member 1 (testing only)                               #
# --------------------------------------------------------------------------- #
def _window(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    pieces, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):  # try to finish on a sentence, else on a space
            cut = text.rfind(". ", start + size // 2, end)
            cut = cut if cut != -1 else text.rfind(" ", start + size // 2, end)
            end = cut + 1 if cut != -1 else end
        pieces.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
        space = text.find(" ", start, end)
        start = space + 1 if space != -1 else start
    return [p for p in pieces if p]


def quick_chunks_from_raw(raw_dir: Path) -> list[Chunk]:
    from pypdf import PdfReader

    chunks: list[Chunk] = []
    for path in sorted(raw_dir.rglob("*")):
        if path.suffix.lower() not in {".pdf", ".docx"}:
            continue
        category = path.parent.name
        if path.suffix.lower() == ".pdf":
            pages = [(i, p.extract_text() or "") for i, p in enumerate(PdfReader(path).pages, start=1)]
        else:
            import docx

            doc = docx.Document(path)
            parts = [p.text for p in doc.paragraphs]
            parts += [cell.text for table in doc.tables for row in table.rows for cell in row.cells]
            pages = [(1, "\n".join(parts))]
        for page_no, text in pages:
            for idx, piece in enumerate(_window(text)):
                chunks.append(Chunk(piece, path.name, category, page_no, idx))
    return chunks


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--chunks", type=Path, help="JSONL file exported by Member 1")
    source.add_argument("--raw", type=Path, help="folder of PDFs/DOCX (quick test loader)")
    parser.add_argument("--reset", action="store_true", help="delete the collection first")
    parser.add_argument("--backend", choices=["minilm", "gemini"], default=None)
    parser.add_argument("--export", type=Path, help="also save the chunks as JSONL")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    chunks = load_jsonl(args.chunks) if args.chunks else quick_chunks_from_raw(args.raw)
    if not chunks:
        sys.exit("No chunks found - check the path.")
    if args.export:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        with args.export.open("w", encoding="utf-8") as fh:
            for c in chunks:
                fh.write(json.dumps({"text": c.text, **c.to_metadata()}, ensure_ascii=False) + "\n")

    store = build_store(backend=args.backend)
    if args.reset:
        store.reset()
    print(f"Embedding {len(chunks)} chunks with {store.embedder.name} ...")
    n = store.add_chunks(chunks)

    print(f"\nUpserted {n} chunks into '{store.collection_name}' at {store.persist_dir}/")
    for category, count in sorted(Counter(c.category for c in chunks).items()):
        print(f"  {category:<28}{count:>4}")
    print(f"Collection now holds {store.count()} vectors.")

    sample = chunks[0]
    vec = store.get_embeddings([sample.chunk_id])[sample.chunk_id]
    print(f"\nExample stored vector  id={sample.chunk_id}")
    print(f"  dimension = {len(vec)},  ||d|| = {magnitude(vec):.4f},  first 5 values = "
          f"{[round(float(x), 4) for x in vec[:5]]}")

    retriever = HybridRetriever(store)
    retriever.refresh_sparse_index()
    print(f"  TF-IDF vocabulary = {retriever.sparse.vocabulary_size} terms over {retriever.sparse.n_docs} chunks")


if __name__ == "__main__":
    main()
