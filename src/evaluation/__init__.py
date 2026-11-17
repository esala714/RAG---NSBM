"""Evaluation of the NSBM RAG system against a hand-checked ground-truth dataset.

ground_truth.json   14 answerable questions (with the document that holds each answer)
                    and 3 questions the documents cannot answer (the agent should refuse)

Run:  python -m src.evaluation.evaluate --help
"""
