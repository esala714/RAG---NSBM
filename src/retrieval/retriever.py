"""Hybrid retriever: dense embeddings (Chroma) + sparse TF-IDF, fused with
Reciprocal Rank Fusion, with the maths behind every score kept in the result.

    RRF(d) = w_dense / (k + rank_dense(d)) + w_sparse / (k + rank_sparse(d))

RRF only uses ranks, so the two very different score scales (embedding
cosine vs TF-IDF cosine) never have to be compared directly. k = 60 is the
value from the original RRF paper (Cormack et al., 2009).

HybridRetriever implements the team interface ``BaseRetriever``
(src/core/interfaces.py), so Member 3's AcademicAgent can use it directly:

    chunks = retriever.search("How late can I enter the exam hall?", top_k=4)
    retriever.last_result.explain()   # the full maths trace of that search

``retrieve()`` returns the richer RetrievalResult for the demos and evaluation:

    result = retriever.retrieve("How late can I enter the exam hall?", k=4)
    result.to_context()      # passages + citations for the prompt
    result.max_dense_cosine  # how confident retrieval is (for refusing)
    result.explain()         # the full maths trace
"""
from __future__ import annotations

import logging
import time

from ..core import models as team
from ..core.interfaces import BaseRetriever
from .schemas import RetrievalResult, RetrievedChunk, ScoreBreakdown
from .sparse_index import TfidfIndex
from .vector_math import explain_cosine
from .vector_store import BaseVectorStore, StoredRecord

logger = logging.getLogger(__name__)


class HybridRetriever(BaseRetriever):
    MODES = ("hybrid", "dense", "sparse")

    def __init__(
        self,
        store: BaseVectorStore,
        sparse_index: TfidfIndex | None = None,
        mode: str = "hybrid",
        rrf_k: int = 60,
        candidate_pool: int = 20,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ):
        self._check_mode(mode)
        self.store = store
        self.sparse = sparse_index or TfidfIndex()
        self.mode = mode
        self.rrf_k = rrf_k
        self.candidate_pool = candidate_pool
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self._records: dict[str, StoredRecord] = {}
        self._indexed_count = -1
        self.last_result: RetrievalResult | None = None  # maths trace of the last search()

    # ------------------------------------------------------------ sparse index
    def refresh_sparse_index(self) -> int:
        """Rebuild TF-IDF from what is stored in Chroma (Chroma is the single
        source of truth, so only one thing has to be persisted)."""
        records = self.store.all_records()
        self._records = {r.chunk_id: r for r in records}
        self.sparse.fit([r.chunk_id for r in records], [r.text for r in records])
        self._indexed_count = len(records)
        logger.info(
            "TF-IDF index built: %d chunks, vocabulary of %d terms",
            self.sparse.n_docs, self.sparse.vocabulary_size,
        )
        return len(records)

    def _ensure_sparse_index(self) -> None:
        if self._indexed_count != self.store.count():
            self.refresh_sparse_index()

    def list_categories(self) -> list[str]:
        """Lets the agent decide whether to narrow a search to one folder."""
        self._ensure_sparse_index()
        return sorted({str(r.metadata.get("category", "general")) for r in self._records.values()})

    # ---------------------------------------------------------------- retrieve
    def retrieve(
        self, question: str, k: int = 4, category: str | None = None, mode: str | None = None
    ) -> RetrievalResult:
        mode = mode or self.mode
        self._check_mode(mode)
        question = (question or "").strip()
        if not question:
            raise ValueError("Question is empty.")

        start = time.perf_counter()
        self._ensure_sparse_index()
        pool = max(k, self.candidate_pool)

        # 1) DENSE: embed the question, nearest neighbours in Chroma (cosine space)
        q_vec = self.store.embedder.embed_query(question)
        dense_hits = self.store.search_by_vector(q_vec, pool, category)
        dense_by_id = {h.chunk_id: h for h in dense_hits}
        dense_rank = {h.chunk_id: i for i, h in enumerate(dense_hits, start=1)}

        # 2) SPARSE: TF-IDF vector of the question, cosine against every chunk
        sq = self.sparse.vectorize(question)
        allowed = None
        if category:
            allowed = {cid for cid, r in self._records.items() if r.metadata.get("category") == category}
        sparse_hits = self.sparse.search(sq, pool, allowed)
        sparse_rank = {h.chunk_id: h.rank for h in sparse_hits}

        # 3) FUSE (mode decides the ordering; both scores are always reported)
        fused = {
            cid: self._rrf(dense_rank.get(cid), sparse_rank.get(cid))
            for cid in set(dense_rank) | set(sparse_rank)
        }
        if mode == "dense":
            ordered = [h.chunk_id for h in dense_hits]
        elif mode == "sparse":
            ordered = [h.chunk_id for h in sparse_hits]
        else:
            ordered = sorted(
                fused, key=lambda cid: (-fused[cid][0], dense_rank.get(cid, 10**9))
            )
        top_ids = ordered[:k]

        # 4) EXPLAIN: full cosine maths for each chunk we return
        missing = [cid for cid in top_ids if cid not in dense_by_id]
        extra_vectors = self.store.get_embeddings(missing)
        chunks: list[RetrievedChunk] = []
        for rank, cid in enumerate(top_ids, start=1):
            hit = dense_by_id.get(cid)
            if hit is not None:
                text, meta, d_vec, distance = hit.text, hit.metadata, hit.embedding, hit.distance
            else:
                record = self._records.get(cid)
                if record is None:
                    continue
                text, meta, d_vec, distance = record.text, record.metadata, extra_vectors.get(cid), None

            cos = explain_cosine(q_vec, d_vec)
            rrf_score, formula = fused.get(cid, (0.0, "0"))
            scores = ScoreBreakdown(
                dot=cos.dot,
                query_norm=cos.norm_a,
                chunk_norm=cos.norm_b,
                dense_cosine=cos.cosine,
                chroma_distance=distance,
                sparse_cosine=self.sparse.score(sq, cid),
                sparse_terms=self.sparse.term_contributions(sq, cid),
                dense_rank=dense_rank.get(cid),
                sparse_rank=sparse_rank.get(cid),
                rrf_score=rrf_score,
                fusion_formula=formula,
            )
            chunks.append(
                RetrievedChunk(
                    rank=rank,
                    chunk_id=cid,
                    text=text,
                    source=str(meta.get("source", "unknown")),
                    category=str(meta.get("category", "general")),
                    page=int(meta.get("page", 0)),
                    scores=scores,
                )
            )
            logger.info(
                "#%d %s p.%s | q.d=%.4f ||q||=%.4f ||d||=%.4f cos=%.4f | tfidf=%.4f | rrf=%.5f",
                rank, meta.get("source"), meta.get("page"), cos.dot, cos.norm_a, cos.norm_b,
                cos.cosine, scores.sparse_cosine, rrf_score,
            )

        return RetrievalResult(
            query=question,
            mode=mode,
            chunks=chunks,
            embedding_model=self.store.embedder.name,
            query_dim=int(len(q_vec)),
            query_norm=explain_cosine(q_vec, q_vec).norm_a,
            query_terms=sq.weights[:8],
            query_sparse_norm=sq.raw_norm,
            unknown_terms=sq.unknown_terms,
            rrf_k=self.rrf_k,
            elapsed_ms=(time.perf_counter() - start) * 1000,
            category_filter=category,
        )

    # ------------------------------------------- team interface (Member 3's agent)
    def search(self, query: str, top_k: int = 5) -> list[team.RetrievedChunk]:
        """BaseRetriever.search(): the "search the documents" tool of the agent.

        The hybrid ranking decides WHICH chunks come back. Each chunk's score is
        its dense cosine with the question (0 to 1), so the agent's min_score
        threshold always means the same thing. All the other numbers (dot
        product, norms, TF-IDF, ranks, RRF) are kept in chunk.extra.
        """
        if not (query or "").strip():
            return []   # e.g. the LLM returned an empty rewrite: nothing to search
        result = self.retrieve(query, k=top_k)
        self.last_result = result
        return [self._to_team_chunk(c, result) for c in result.chunks]

    @staticmethod
    def _to_team_chunk(c: RetrievedChunk, result: RetrievalResult) -> team.RetrievedChunk:
        s = c.scores
        return team.RetrievedChunk(
            text=c.text,
            source=c.source,
            page=c.page if c.page > 0 else None,     # DOCX / TXT files have no pages
            score=round(min(1.0, max(0.0, s.dense_cosine)), 6),
            extra={
                "chunk_id": c.chunk_id,
                "category": c.category,
                "rank": c.rank,
                "mode": result.mode,
                "embedding_model": result.embedding_model,
                "dot": round(s.dot, 6),
                "query_norm": round(s.query_norm, 6),
                "chunk_norm": round(s.chunk_norm, 6),
                "dense_cosine": round(s.dense_cosine, 6),
                "chroma_distance": None if s.chroma_distance is None else round(float(s.chroma_distance), 6),
                "sparse_cosine": round(float(s.sparse_cosine), 6),
                "sparse_terms": [[t, round(float(w), 6)] for t, w in s.sparse_terms],
                "dense_rank": s.dense_rank,
                "sparse_rank": s.sparse_rank,
                "rrf_score": round(float(s.rrf_score), 6),
                "fusion_formula": s.fusion_formula,
            },
        )

    # ----------------------------------------------------------------- helpers
    def _rrf(self, d_rank: int | None, s_rank: int | None) -> tuple[float, str]:
        score, parts = 0.0, []
        for weight, rank in ((self.dense_weight, d_rank), (self.sparse_weight, s_rank)):
            if rank is None or weight == 0:
                continue
            score += weight / (self.rrf_k + rank)
            w = "1" if weight == 1 else f"{weight:g}"
            parts.append(f"{w}/({self.rrf_k}+{rank})")
        return score, " + ".join(parts) or "0"

    def _check_mode(self, mode: str) -> None:
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got '{mode}'")
