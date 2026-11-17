"""Tests for the retrieval layer.   Run:  pytest -q tests/test_retrieval.py

A tiny deterministic FakeEmbedder is injected instead of MiniLM, so the
tests run in seconds with no model download and no API key - this is the
practical benefit of dependency injection.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest

from src.retrieval.schemas import Chunk
from src.retrieval.sparse_index import TfidfIndex, tokenize
from src.retrieval.vector_math import cosine, dot, explain_cosine, magnitude, normalize

chromadb = pytest.importorskip("chromadb")

from src.retrieval.embedders import BaseEmbedder  # noqa: E402
from src.retrieval.retriever import HybridRetriever  # noqa: E402
from src.retrieval.vector_store import ChromaVectorStore, EmbeddingMismatchError  # noqa: E402


class FakeEmbedder(BaseEmbedder):
    """Hashed bag-of-words -> 64-d vector. Same words -> similar vectors."""

    def __init__(self, name: str = "fake-64d", dim: int = 64):
        self._name, self._dim = name, dim

    @property
    def name(self) -> str:
        return self._name

    def embed_documents(self, texts):
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in tokenize(text):
                out[row, int(hashlib.md5(tok.encode()).hexdigest(), 16) % self._dim] += 1.0
        return out


CHUNKS = [
    Chunk("Candidates shall be admitted to the examination room up to thirty minutes after the start.",
          "Examination Regulations.pdf", "Exam", 1, 0),
    Chunk("A student must have a minimum GPA of 3.70 to be awarded First Class Honours.",
          "Exam By Laws.pdf", "GPA", 4, 0),
    Chunk("Permanent staff are entitled to fourteen days of annual leave and seven days of casual leave.",
          "NSBM_Leave_Policy.pdf", "leave_policy", 2, 0),
    Chunk("New employees complete an orientation programme during their first week of onboarding.",
          "NSBM_Employee_Onbording.pdf", "onboarding", 1, 0),
]


@pytest.fixture()
def store(tmp_path):
    s = ChromaVectorStore(FakeEmbedder(), persist_dir=tmp_path / "db", collection_name="test")
    s.add_chunks(CHUNKS)
    return s


# ----------------------------------------------------------------- vector maths
def test_vector_math_known_values():
    assert magnitude([3, 4]) == pytest.approx(5.0)
    assert dot([1, 2, 3], [4, 5, 6]) == pytest.approx(32.0)
    assert magnitude(normalize([3, 4])) == pytest.approx(1.0)
    assert cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine([2, 2], [5, 5]) == pytest.approx(1.0)       # same direction, any length
    e = explain_cosine([1, 2], [2, 3])
    assert e.cosine == pytest.approx(8 / (math.sqrt(5) * math.sqrt(13)))


# ---------------------------------------------------------------------- TF-IDF
def test_tfidf_rows_are_unit_length_and_rare_terms_weigh_more():
    idx = TfidfIndex().fit([c.chunk_id for c in CHUNKS], [c.text for c in CHUNKS])
    assert np.allclose(np.linalg.norm(idx.matrix, axis=1), 1.0, atol=1e-5)
    # "leave" occurs in 1 of 4 chunks, so its idf is above the minimum possible value (1.0)
    assert idx.idf[idx.vocabulary["leave"]] > 1.0


def test_tfidf_cosine_equals_sum_of_term_products():
    idx = TfidfIndex().fit([c.chunk_id for c in CHUNKS], [c.text for c in CHUNKS])
    q = idx.vectorize("how many days of annual leave")
    best = idx.search(q, k=1)[0]
    assert best.chunk_id == CHUNKS[2].chunk_id
    contributions = idx.term_contributions(q, best.chunk_id, top=100)
    assert sum(w for _, w in contributions) == pytest.approx(best.cosine, abs=1e-5)


# ----------------------------------------------------------------- vector store
def test_upsert_is_idempotent(store):
    assert store.count() == 4
    store.add_chunks(CHUNKS)              # running the indexer twice must not duplicate
    assert store.count() == 4


def test_data_persists_across_new_store_objects(store, tmp_path):
    reopened = ChromaVectorStore(FakeEmbedder(), persist_dir=tmp_path / "db", collection_name="test")
    assert reopened.count() == 4
    assert set(reopened.list_categories()) == {"Exam", "GPA", "leave_policy", "onboarding"}


def test_wrong_embedder_is_refused(store, tmp_path):
    with pytest.raises(EmbeddingMismatchError):
        ChromaVectorStore(FakeEmbedder(name="other-model"), persist_dir=tmp_path / "db",
                          collection_name="test")


def test_metadata_is_stored(store):
    record = {r.chunk_id: r for r in store.all_records()}[CHUNKS[1].chunk_id]
    assert record.metadata["source"] == "Exam By Laws.pdf"
    assert record.metadata["category"] == "GPA"
    assert record.metadata["page"] == 4


def test_chunk_accepts_member1_dict_shapes():
    nested = Chunk.from_any({"page_content": "x y z", "metadata": {"document_name": "a.pdf",
                                                                     "category": "Exam", "page_number": 3}})
    assert (nested.source, nested.category, nested.page) == ("a.pdf", "Exam", 3)
    flat = Chunk.from_any({"text": "x y z", "source": "a.pdf", "category": "Exam", "page": 3})
    assert flat == nested


def test_chunk_reads_member1_jsonl_record():
    """The exact shape written by src/ingestion/document_processor.py."""
    record = {
        "id": "Repeat coursework Procedure-2-1",
        "text": "Submit your assignment within 14 days of downloading it.",
        "metadata": {"source": "Repeat coursework Procedure.pdf", "doc_title": "Repeat coursework Procedure",
                     "category": "examinations", "file_type": "pdf", "page_number": 2,
                     "chunk_index": 1, "word_count": 9, "doc_hash": "abc123"},
    }
    chunk = Chunk.from_any(record)
    assert (chunk.source, chunk.category, chunk.page, chunk.chunk_index) == (
        "Repeat coursework Procedure.pdf", "examinations", 2, 1)


def test_file_without_pages_is_cited_without_a_page(tmp_path):
    """Member 1 writes page_number = None for DOCX/TXT files."""
    docx_chunk = Chunk.from_any({"text": "Library opening hours are listed on the portal.",
                                 "metadata": {"source": "Guide.docx", "category": "general",
                                              "page_number": None, "chunk_index": 1}})
    assert docx_chunk.page == 0
    store = ChromaVectorStore(FakeEmbedder(), persist_dir=tmp_path / "db", collection_name="nopage")
    store.add_chunks([docx_chunk])
    result = HybridRetriever(store).retrieve("library opening hours", k=1)
    assert result.chunks[0].citation == "Guide.docx"
    assert "page" not in result.to_context().splitlines()[0]


# -------------------------------------------------------------------- retriever
@pytest.mark.parametrize("mode", ["hybrid", "dense", "sparse"])
def test_retrieve_finds_the_right_chunk(store, mode):
    result = HybridRetriever(store, mode=mode).retrieve("annual leave days for staff", k=2)
    assert result.chunks[0].source == "NSBM_Leave_Policy.pdf"
    assert result.chunks[0].page == 2


def test_our_cosine_matches_chromas_distance(store):
    result = HybridRetriever(store, mode="dense").retrieve("First Class Honours GPA", k=3)
    for c in result.chunks:
        assert c.scores.chroma_distance is not None
        assert c.scores.dense_cosine == pytest.approx(1 - c.scores.chroma_distance, abs=1e-4)


def test_category_filter(store):
    result = HybridRetriever(store).retrieve("days", k=4, category="leave_policy")
    assert result.chunks and all(c.category == "leave_policy" for c in result.chunks)


def test_trace_and_context_are_printable(store):
    result = HybridRetriever(store).retrieve("examination room thirty minutes", k=2)
    text = result.explain()
    assert "cos = q.d / (||q|| x ||d||)" in text and "RRF" in text
    assert "[1] Source: Examination Regulations.pdf (page 1" in result.to_context()
    assert 0.0 < result.max_dense_cosine <= 1.0
