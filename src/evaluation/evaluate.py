"""Evaluate the NSBM RAG system on the ground-truth questions and print a per-question report.

    python -m src.evaluation.evaluate                                  offline TF-IDF baseline
    python -m src.evaluation.evaluate --retriever hybrid               Member 2's Chroma store (build it first)
    python -m src.evaluation.evaluate --retriever hybrid --with-agent  full RAG: needs GEMINI_API_KEY in .env

The full results (every retrieved chunk, every answer sentence and its support score) are
saved as JSON in logs/evaluation/. The formulas are in src/evaluation/metrics.py.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.core.interfaces import BaseRetriever
from src.evaluation.metrics import (
    answer_chunk_rank,
    answer_relevance,
    context_recall,
    context_relevance,
    groundedness,
    hit_at_k,
    mean,
    reciprocal_rank,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GROUND_TRUTH = PROJECT_ROOT / "src" / "evaluation" / "ground_truth.json"
DEFAULT_CHUNKS = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "logs" / "evaluation"
LINE = "-" * 78


def load_ground_truth(path: str | Path = DEFAULT_GROUND_TRUTH) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as file:
        questions = json.load(file)
    for q in questions:
        if q["answerable"] and not q["expected_sources"]:
            raise ValueError(f"{q['id']}: an answerable question needs expected_sources.")
        if q["answerable"] and not q.get("context"):
            raise ValueError(f"{q['id']}: an answerable question needs its context passage.")
    return questions


# --------------------------------------------------------------------------- #
# 1. Retrieval                                                                #
# --------------------------------------------------------------------------- #
def evaluate_retrieval(retriever: BaseRetriever, questions: list[dict], k: int = 4) -> dict:
    rows = []
    for q in questions:
        chunks = retriever.search(q["question"], k)
        row: dict[str, Any] = {
            "id": q["id"],
            "question": q["question"],
            "answerable": q["answerable"],
            "top_score": round(max((c.score for c in chunks), default=0.0), 4),
            "retrieved": [
                {"rank": i, "source": c.source, "page": c.page, "score": round(c.score, 4)}
                for i, c in enumerate(chunks, start=1)
            ],
        }
        if q["answerable"]:
            sources = [c.source for c in chunks]
            row["hit"] = hit_at_k(sources, q["expected_sources"])
            row["reciprocal_rank"] = round(reciprocal_rank(sources, q["expected_sources"]), 4)
            row["context_relevance"] = round(
                context_relevance([c.text for c in chunks], q["relevant_terms"]), 4)
            row["context_recall"] = (round(context_recall([c.text for c in chunks], q["context"]), 4)
                                     if q.get("context") else None)
            rank = answer_chunk_rank([c.text for c in chunks], q["context"]) if q.get("context") else None
            row["answer_chunk_rank"] = rank
            row["chunk_hit"] = (1.0 if rank else 0.0) if q.get("context") else None
            row["chunk_reciprocal_rank"] = (round(1.0 / rank, 4) if rank else 0.0) if q.get("context") else None
        rows.append(row)

    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    summary = {
        "k": k,
        "answerable_questions": len(answerable),
        "unanswerable_questions": len(unanswerable),
        "hits": int(sum(r["hit"] for r in answerable)),
        "hit_rate": mean(r["hit"] for r in answerable),
        "mrr": mean(r["reciprocal_rank"] for r in answerable),
        "chunk_hits": int(sum(r["chunk_hit"] or 0 for r in answerable)),
        "chunk_hit_rate": mean(r["chunk_hit"] for r in answerable),
        "chunk_mrr": mean(r["chunk_reciprocal_rank"] for r in answerable),
        "context_relevance": mean(r["context_relevance"] for r in answerable),
        "context_recall": mean(r["context_recall"] for r in answerable),
        # Compare these two to choose the agent's min_score: a good threshold lies between them.
        "lowest_top_score_answerable": min((r["top_score"] for r in answerable), default=None),
        "highest_top_score_unanswerable": max((r["top_score"] for r in unanswerable), default=None),
    }
    return {"summary": summary, "questions": rows}


# --------------------------------------------------------------------------- #
# 2. Answers (agent + LLM)                                                    #
# --------------------------------------------------------------------------- #
def _llm_failed(response: Any) -> bool:
    return any(step.detail.startswith("Gemini unavailable") for step in response.trace)


def evaluate_agent(agent: Any, questions: list[dict]) -> dict:
    """`agent` is anything with run(question) -> RAGResponse, e.g. Member 3's AcademicAgent."""
    rows = []
    for q in questions:
        response = agent.run(q["question"])
        refused = not response.answerable
        row: dict[str, Any] = {
            "id": q["id"],
            "question": q["question"],
            "answerable": q["answerable"],
            "answer": response.answer,
            "refused": refused,
            "llm_error": _llm_failed(response),
            "sources": [f"{c.source}, page {c.page}" if c.page else c.source for c in response.sources],
            "groundedness": None,
            "sentences": [],
        }
        if not refused:
            score, details = groundedness(response.answer, [c.text for c in response.sources])
            row["groundedness"] = None if score is None else round(score, 4)
            row["sentences"] = details
        if q["answerable"]:
            row["answer_relevance"] = 0.0 if refused else round(
                answer_relevance(response.answer, q["expected_keywords"]), 4)
        else:
            row["correct_refusal"] = refused
        rows.append(row)

    usable = [r for r in rows if not r["llm_error"]]
    answerable = [r for r in usable if r["answerable"]]
    unanswerable = [r for r in usable if not r["answerable"]]
    summary = {
        "questions_scored": len(usable),
        "llm_errors": len(rows) - len(usable),
        "groundedness": mean(r["groundedness"] for r in usable),
        "answer_relevance": mean(r["answer_relevance"] for r in answerable),
        "false_refusal_rate": mean(float(r["refused"]) for r in answerable),
        "refusal_accuracy": mean(float(r["correct_refusal"]) for r in unanswerable),
    }
    return {"summary": summary, "questions": rows}


# --------------------------------------------------------------------------- #
# Report                                                                      #
# --------------------------------------------------------------------------- #
def _ascii(text: str) -> str:
    """Windows terminals cannot always print curly quotes or dashes."""
    return str(text).encode("ascii", "replace").decode("ascii")


def _fmt(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def format_retrieval_report(result: dict, retriever_name: str) -> str:
    s = result["summary"]
    lines = [
        f"RETRIEVAL  |  retriever: {retriever_name}  |  k = {s['k']}  |  "
        f"{s['answerable_questions']} answerable + {s['unanswerable_questions']} unanswerable questions",
        LINE,
        f"{'ID':<4} {'Hit':>3} {'RR':>5} {'CHit':>4} {'CRR':>5} {'CtxRel':>6} {'Recall':>6} {'Top':>6}  Best chunk",
    ]
    for r in result["questions"]:
        best = r["retrieved"][0] if r["retrieved"] else None
        if best is None:
            where = "(no chunk)"
        else:   # DOCX / TXT chunks have no page number
            where = best["source"] + (f" (p{best['page']})" if best["page"] else "")
        if r["answerable"]:
            cells = (f"{int(r['hit']):>3} {r['reciprocal_rank']:>5.2f} {_fmt(r['chunk_hit'], 0):>4}"
                     f" {_fmt(r['chunk_reciprocal_rank'], 2):>5} {r['context_relevance']:>6.2f}"
                     f" {_fmt(r['context_recall'], 2):>6}")
        else:
            cells = f"{'-':>3} {'-':>5} {'-':>4} {'-':>5} {'-':>6} {'-':>6}"
        lines.append(_ascii(f"{r['id']:<4} {cells} {r['top_score']:>6.3f}  {where[:45]}"))
    lines += [
        LINE,
        f"Hit@{s['k']}              = {s['hits']}/{s['answerable_questions']} = {_fmt(s['hit_rate'])}",
        f"MRR                = {_fmt(s['mrr'])}",
        f"Chunk Hit@{s['k']}        = {s['chunk_hits']}/{s['answerable_questions']} = {_fmt(s['chunk_hit_rate'])}"
        "   (a retrieved chunk holds the answer, not only the right document)",
        f"Chunk MRR          = {_fmt(s['chunk_mrr'])}",
        f"Context Relevance  = {_fmt(s['context_relevance'])}",
        f"Context Recall     = {_fmt(s['context_recall'])}",
        f"Top score: lowest for an answerable question = {_fmt(s['lowest_top_score_answerable'])}, "
        f"highest for an unanswerable question = {_fmt(s['highest_top_score_unanswerable'])}",
    ]
    return "\n".join(lines)


def format_agent_report(result: dict) -> str:
    s = result["summary"]
    lines = ["ANSWERS (agent + Gemini)", LINE,
             f"{'ID':<4} {'Refused':>7} {'Ground':>6} {'AnsRel':>6}  Answer"]
    for r in result["questions"]:
        refused = "error" if r["llm_error"] else ("yes" if r["refused"] else "no")
        relevance = _fmt(r.get("answer_relevance"), 2) if r["answerable"] else "-"
        answer = " ".join(r["answer"].split())[:44]
        lines.append(_ascii(f"{r['id']:<4} {refused:>7} {_fmt(r['groundedness'], 2):>6} {relevance:>6}  {answer}"))
    lines += [
        LINE,
        f"Groundedness       = {_fmt(s['groundedness'])}",
        f"Answer Relevance   = {_fmt(s['answer_relevance'])}",
        f"Refusal accuracy   = {_fmt(s['refusal_accuracy'])}  (unanswerable questions correctly refused)",
        f"False refusal rate = {_fmt(s['false_refusal_rate'])}  (answerable questions wrongly refused)",
    ]
    if s["llm_errors"]:
        lines.append(f"Skipped {s['llm_errors']} question(s) because Gemini was unavailable.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Command line                                                                #
# --------------------------------------------------------------------------- #
def build_retriever(name: str, chunks_path: Path, mode: str | None) -> BaseRetriever:
    if name == "tfidf":
        from src.evaluation.retrievers import TfidfRetriever

        return TfidfRetriever.from_jsonl(chunks_path)
    # Member 2's HybridRetriever implements BaseRetriever (score = dense cosine).
    # It needs chromadb and a store built with scripts/build_index.py.
    from src.retrieval import build_retriever as build_hybrid

    return build_hybrid(mode=mode)


def build_agent(retriever: BaseRetriever, k: int, min_score: float | None):
    import os

    from src.agent.academic_agent import AcademicAgent
    from src.rag.llm_client import GeminiClient
    from src.rag.prompt_builder import PromptBuilder

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    api_key, model = os.getenv("GEMINI_API_KEY"), os.getenv("GEMINI_MODEL")
    if not api_key or not model:
        raise SystemExit("--with-agent needs GEMINI_API_KEY and GEMINI_MODEL in .env (see env.example).")
    settings = {"top_k": k, "trace_dir": str(PROJECT_ROOT / "logs" / "traces")}
    if min_score is not None:
        settings["min_score"] = min_score
    return AcademicAgent(retriever, GeminiClient(api_key, model), PromptBuilder(), **settings)


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Evaluate the NSBM RAG system.")
    parser.add_argument("--retriever", choices=["tfidf", "hybrid"], default="tfidf",
                        help="tfidf: offline baseline over chunks.jsonl; hybrid: Member 2's Chroma store")
    parser.add_argument("--mode", choices=["hybrid", "dense", "sparse"], default=None,
                        help="search mode for --retriever hybrid (default: RETRIEVAL_MODE or hybrid)")
    parser.add_argument("--k", type=int, default=4, help="chunks retrieved per question")
    parser.add_argument("--with-agent", action="store_true",
                        help="also run Member 3's agent with Gemini and score its answers")
    parser.add_argument("--min-score", type=float, default=None,
                        help="agent's similarity threshold (default 0.35; TF-IDF scores are lower, try 0.1)")
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--output", type=Path, default=None, help="JSON results file")
    args = parser.parse_args(argv)

    questions = load_ground_truth(args.ground_truth)
    retriever = build_retriever(args.retriever, args.chunks, args.mode)

    results: dict[str, Any] = {
        "settings": {"retriever": args.retriever, "mode": args.mode, "k": args.k,
                     "questions": len(questions), "run_at": datetime.now().isoformat(timespec="seconds")},
        "retrieval": evaluate_retrieval(retriever, questions, args.k),
    }
    print(format_retrieval_report(results["retrieval"], args.retriever))

    if args.with_agent:
        agent = build_agent(retriever, args.k, args.min_score)
        results["agent"] = evaluate_agent(agent, questions)
        print()
        print(format_agent_report(results["agent"]))

    suffix = "_agent" if args.with_agent else ""
    output = args.output or DEFAULT_OUTPUT_DIR / f"results_{args.retriever}{suffix}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nFull results saved to {output}")
    return results


if __name__ == "__main__":
    main()
