"""Ask the vector store a question and print the retrieved chunks with their maths.

    python scripts/search_demo.py "How late can I enter the exam hall?"
    python scripts/search_demo.py "First class GPA" --mode dense --k 3
    python scripts/search_demo.py "annual leave days" --category leave_policy
    python scripts/search_demo.py                      # interactive loop
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retrieval import build_retriever  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", nargs="?", help="leave empty for interactive mode")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--mode", choices=["hybrid", "dense", "sparse"], default=None)
    parser.add_argument("--category", default=None, help="e.g. Exam, GPA, leave_policy")
    parser.add_argument("--backend", choices=["minilm", "gemini"], default=None)
    parser.add_argument("--json", action="store_true", help="print the raw result as JSON")
    parser.add_argument("--log", action="store_true", help="also show INFO logs of each score")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.log else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    retriever = build_retriever(backend=args.backend, mode=args.mode)
    if retriever.store.count() == 0:
        sys.exit("The vector store is empty - run scripts/build_index.py first.")
    print(f"Collection holds {retriever.store.count()} chunks. Categories: "
          f"{', '.join(retriever.list_categories())}\n")

    def ask(question: str) -> None:
        result = retriever.retrieve(question, k=args.k, category=args.category)
        print(json.dumps(result.to_dict(), indent=2) if args.json else result.explain())

    if args.question:
        ask(args.question)
        return
    while True:
        question = input("\nQuestion (blank to quit): ").strip()
        if not question:
            break
        ask(question)


if __name__ == "__main__":
    main()
