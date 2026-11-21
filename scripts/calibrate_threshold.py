"""Choose the agent's refusal threshold (min_score) from the hybrid retriever's scores.

The agent answers only when a retrieved chunk scores at least min_score (the dense
cosine that HybridRetriever.search() returns) and refuses otherwise. This script
asks every evaluation question through search(), exactly as the agent does, and
picks the threshold that best separates the questions the documents can answer
from those they cannot (the method is in src/retrieval/threshold.py).

Questions: Member 1's ground truth (src/evaluation/ground_truth.json), plus the
extra "near miss" questions below, which the documents cannot answer either.
Three unanswerable questions are too few to place a threshold on their own.

    python scripts/build_index.py --reset           (once, after the chunks change)
    python scripts/calibrate_threshold.py           (mode from RETRIEVAL_MODE, default hybrid)
    python scripts/calibrate_threshold.py --mode dense
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.evaluate import DEFAULT_GROUND_TRUTH, load_ground_truth  # noqa: E402
from src.retrieval import build_retriever  # noqa: E402
from src.retrieval.threshold import balanced_accuracy, choose_threshold, outcome, sweep  # noqa: E402

AGENT_DEFAULT = 0.35   # AcademicAgent(min_score=0.35)

# Near misses: about the university, but not in any document (checked by searching the chunks).
EXTRA_UNANSWERABLE = [
    ("x01", "What is the WiFi password in the library?"),
    ("x02", "Is there a shuttle bus from Colombo to the campus?"),
    ("x03", "What is the dress code for the graduation ceremony?"),
    ("x04", "How do I apply for a student visa to study in Australia?"),
    ("x05", "What is the monthly salary of a lecturer at NSBM?"),
    ("x06", "Does NSBM offer sports scholarships?"),
    ("x07", "What is the capital city of France?"),
]

LINE = "-" * 78


def ascii_only(text: str) -> str:
    """Windows terminals cannot always print curly quotes or dashes."""
    return str(text).encode("ascii", "replace").decode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["hybrid", "dense", "sparse"], default=None,
                        help="search mode (default: RETRIEVAL_MODE or hybrid)")
    parser.add_argument("--k", type=int, default=4, help="chunks per search, as the agent's top_k")
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    args = parser.parse_args()

    retriever = build_retriever(mode=args.mode)
    if retriever.store.count() == 0:
        sys.exit("The vector store is empty. Run: python scripts/build_index.py --reset")

    questions = [(q["id"], q["question"], q["answerable"], q.get("expected_sources", []))
                 for q in load_ground_truth(args.ground_truth)]
    questions += [(qid, text, False, []) for qid, text in EXTRA_UNANSWERABLE]

    print(f"Mode = {retriever.mode}, k = {args.k}, {retriever.store.count()} chunks, "
          f"model = {retriever.store.embedder.name}")
    print("Score = best dense cosine of the k chunks, the number the agent compares with min_score")
    print(LINE)
    print(f"{'ID':<4} {'Group':<12} {'Score':>6} {'Right doc':>9}  Question")

    answerable, unanswerable, rows = [], [], []
    for qid, text, can_answer, expected in questions:
        chunks = retriever.search(text, top_k=args.k)
        score = max((c.score for c in chunks), default=0.0)
        if can_answer:
            answerable.append(score)
            found = "yes" if any(c.source in expected for c in chunks) else "NO"
        else:
            unanswerable.append(score)
            found = "-"
        rows.append((qid, text, can_answer, score))
        group = "answerable" if can_answer else "unanswerable"
        print(ascii_only(f"{qid:<4} {group:<12} {score:>6.3f} {found:>9}  {text[:46]}"))

    print(LINE)
    print(f"Lowest answerable score   = {min(answerable):.3f}")
    print(f"Highest unanswerable score = {max(unanswerable):.3f}")
    if min(answerable) > max(unanswerable):
        print("The two groups do not overlap: one threshold can separate them all.")
    else:
        print("The two groups overlap: some questions are misjudged at any threshold.")

    n_ans, n_unans = len(answerable), len(unanswerable)
    print(f"\n{'min_score':>9}  answered (of {n_ans})  refused (of {n_unans})  balanced accuracy")
    thresholds = [round(0.20 + 0.05 * i, 2) for i in range(9)]
    for t, answered, refused in sweep(answerable, unanswerable, thresholds):
        note = "   <- agent default" if t == AGENT_DEFAULT else ""
        bal = balanced_accuracy(answered, refused, n_ans, n_unans)
        print(f"{t:>9.2f}  {answered:>19}  {refused:>18}  {bal:>17.3f}{note}")

    choice = choose_threshold(answerable, unanswerable)
    suggested = round(choice.threshold, 2)
    answered, refused = outcome(suggested, answerable, unanswerable)
    print(LINE)
    print(f"Best threshold (middle of the widest gap with the best balanced accuracy) = {choice.threshold:.3f}, "
          f"margin {choice.margin:.3f} to the nearest score")
    print(f"Best min_score on these scores = {suggested:.2f}: {answered}/{n_ans} answerable answered, "
          f"{refused}/{n_unans} unanswerable refused, "
          f"balanced accuracy {balanced_accuracy(answered, refused, n_ans, n_unans):.3f}")

    print("Check it end to end before changing the agent: "
          f"python -m src.evaluation.evaluate --retriever hybrid --with-agent --min-score {suggested:.2f}")

    wrong = [(qid, text, ok, s) for qid, text, ok, s in rows if (s >= suggested) != ok]
    for qid, text, can_answer, score in wrong:
        what = "refused although answerable" if can_answer else "answered although unanswerable"
        print(ascii_only(f"  misjudged: {qid} ({score:.3f}) {what}: {text[:40]}"))


if __name__ == "__main__":
    main()
