"""Build (or update) the persistent Chroma index from Member 1's chunks.

Normal use - Member 1's document processor writes one chunk per line to
data/processed/chunks.jsonl:
    {"id": "...", "text": "...", "metadata": {"source": "Repeat coursework Procedure.pdf",
     "category": "examinations", "page_number": 2, "chunk_index": 1, ...}}

    python scripts/build_index.py --reset                 (reads data/processed/chunks.jsonl)
    python scripts/build_index.py --chunks other.jsonl --reset

Straight from the documents - runs Member 1's src/ingestion/document_processor.py
on every file, or a simple stand-in loader if that module is not in the repo yet:

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

DEFAULT_CHUNKS = ROOT / "data" / "processed" / "chunks.jsonl"
WORDS_PER_TOKEN = 0.75  # an English word is about 1.3 word-piece tokens


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


def chunks_from_raw(raw_dir: Path) -> list[Chunk]:
    """Use Member 1's processor when it is in the repo, else the stand-in loader."""
    try:
        from src.ingestion import document_processor as member1
    except ImportError:
        print("Member 1's src/ingestion/document_processor.py not found - using the simple stand-in loader.")
        return quick_chunks_from_raw(raw_dir)
    print("Chunking with Member 1's document processor ...")
    chunks: list[Chunk] = []
    for path in sorted(raw_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in member1.SUPPORTED:
            chunks.extend(Chunk.from_any(record) for record in member1.process_file(path))
    return chunks


# --------------------------------------------------------------------------- #
# Stand-in loader, only used when Member 1's module is missing                 #
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
            pages = [(0, "\n".join(parts))]
        for page_no, text in pages:
            for idx, piece in enumerate(_window(text)):
                chunks.append(Chunk(piece, path.name, category, page_no, idx))
    return chunks


# --------------------------------------------------------------------------- #
def report_quality(chunks: list[Chunk], unique: list[Chunk], max_tokens: int | None) -> None:
    """Point out problems in the input that change retrieval quality."""
    duplicates = len(chunks) - len(unique)
    if duplicates:
        repeated = Counter(c.source for c in chunks) - Counter(c.source for c in unique)
        print(f"  {duplicates} exact duplicate chunks skipped "
              f"(same file, page and text): {', '.join(sorted(repeated))}")
    if max_tokens:
        word_limit = int(max_tokens * WORDS_PER_TOKEN)
        too_long = [c for c in unique if len(c.text.split()) > word_limit]
        if too_long:
            print(f"  WARNING: {len(too_long)} of {len(unique)} chunks are longer than ~{word_limit} words. "
                  f"The embedding model reads only the first {max_tokens} tokens (~{word_limit} words) of a chunk,\n"
                  f"  so the rest of those chunks is invisible to dense search (TF-IDF still sees every word). "
                  f"Smaller chunks (about {word_limit - 40}-{word_limit} words) would fix this.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--chunks", type=Path, help=f"JSONL file from Member 1 (default: {DEFAULT_CHUNKS})")
    source.add_argument("--raw", type=Path, help="folder of documents to chunk first")
    parser.add_argument("--reset", action="store_true", help="delete the collection first")
    parser.add_argument("--backend", choices=["minilm", "gemini"], default=None)
    parser.add_argument("--export", type=Path, help="also save the chunks as JSONL")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    if args.raw:
        chunks = chunks_from_raw(args.raw)
    else:
        path = args.chunks or DEFAULT_CHUNKS
        if not path.exists():
            sys.exit(f"{path} not found. Run Member 1's document processor first, or use --raw data/raw.")
        chunks = load_jsonl(path)
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
    unique = list({c.chunk_id: c for c in chunks}.values())
    print(f"Read {len(chunks)} chunks ({len(unique)} unique). Embedding with {store.embedder.name} ...")
    report_quality(chunks, unique, store.embedder.max_tokens)
    n = store.add_chunks(unique)

    print(f"\nUpserted {n} chunks into '{store.collection_name}' at {store.persist_dir}/")
    for category, count in sorted(Counter(c.category for c in unique).items()):
        print(f"  {category:<28}{count:>4}")
    print(f"Collection now holds {store.count()} vectors.")

    sample = unique[0]
    vec = store.get_embeddings([sample.chunk_id])[sample.chunk_id]
    print(f"\nExample stored vector  id={sample.chunk_id}")
    print(f"  dimension = {len(vec)},  ||d|| = {magnitude(vec):.4f},  first 5 values = "
          f"{[round(float(x), 4) for x in vec[:5]]}")

    retriever = HybridRetriever(store)
    retriever.refresh_sparse_index()
    print(f"  TF-IDF vocabulary = {retriever.sparse.vocabulary_size} terms over {retriever.sparse.n_docs} chunks")


if __name__ == "__main__":
    main()
