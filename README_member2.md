# Member 2: Embeddings and Vector Database

This module takes the chunks from Member 1, embeds them, stores them in a **persistent Chroma DB**, and returns the most relevant chunks for a question. It also returns **the maths behind each score**: dot product, magnitudes, cosine, TF-IDF weights and the rank fusion.

```
Member 1 chunks ──► Embedder (MiniLM / Gemini) ──► ChromaVectorStore (on disk, cosine space)
                                                          │
question ──► HybridRetriever ──► dense search (Chroma) ───┤
                     │──────────► sparse search (TF-IDF) ─┴─► RRF fusion ──► RetrievalResult ──► Member 3
```

## Files

| File | What it contains |
|---|---|
| `src/retrieval/schemas.py` | `Chunk` (input from Member 1) and `RetrievalResult` (output to Member 3), plus `explain()` and `to_context()` |
| `src/retrieval/vector_math.py` | Magnitude, dot product, normalisation and cosine, written out explicitly |
| `src/retrieval/embedders.py` | `BaseEmbedder` (ABC), `MiniLMEmbedder` (default, free, offline), `GeminiEmbedder` (optional) |
| `src/retrieval/vector_store.py` | `BaseVectorStore` (ABC) and `ChromaVectorStore` (persistent, upsert, metadata, cosine search) |
| `src/retrieval/sparse_index.py` | TF-IDF vector space model built from scratch with NumPy |
| `src/retrieval/retriever.py` | `HybridRetriever` (implements the team's `BaseRetriever`): dense + sparse + Reciprocal Rank Fusion, with the score trace |
| `src/retrieval/factory.py` | The composition root, `build_retriever()`, which reads settings from the environment |
| `scripts/build_index.py` | Indexes Member 1's `data/processed/chunks.jsonl` (default), or runs Member 1's processor on `data/raw` |
| `scripts/search_demo.py` | Prints retrieved chunks and their maths (use this for the video) |
| `tests/test_retrieval.py` | 20 pytest tests that run offline with a fake embedder, including the agent running on the real retriever |

## Interface with the team

**Input from Member 1** is `data/processed/chunks.jsonl`, written by `src/ingestion/document_processor.py`. Each line is one chunk:

```json
{"id": "NSBM academic appeals policy-2-1",
 "text": "1. Valid Grounds for a Results Appeal ...",
 "metadata": {"source": "NSBM academic appeals policy.pdf", "doc_title": "NSBM academic appeals policy",
              "category": "academic_policies", "file_type": "pdf", "page_number": 2,
              "chunk_index": 1, "word_count": 289, "doc_hash": "a418c6..."}}
```

What Member 2 reads from each record:

| Field | Used as | Notes |
|---|---|---|
| `text` | the passage that is embedded and returned | |
| `metadata.source` | citation | file name |
| `metadata.category` | filter (`category=`) | e.g. `academic_policies`, `examinations`, `registration`, `attendance` |
| `metadata.page_number` | citation | 1-based for PDFs; `null` for DOCX/TXT, stored as 0 and cited without a page |
| `metadata.chunk_index` | part of the chunk id | |

Member 1's `id` is not used: Member 2 makes its own id from source, page, chunk index and text, so identical chunks are stored once. Flat records (`{"text", "source", "category", "page"}`) and the names `page_content`/`content`/`document_name`/`file_name` are accepted too.

```bash
python scripts/build_index.py --reset                 # reads data/processed/chunks.jsonl
python scripts/build_index.py --raw data/raw --reset  # runs Member 1's processor on the raw files
```

`build_index.py` also checks the chunks before indexing:

* **Duplicates:** exact copies (same file, page and text) are stored once, and the script names the files they came from.
* **Length:** MiniLM reads only the first 256 word-pieces (about 190 words) of a chunk and ignores the rest. The script warns when chunks are longer than that, so Member 1 can lower `chunk_size`. Gemini embeddings read up to 2048 tokens.

**Output to Member 3 (RAG and agent):** `HybridRetriever` inherits the team interface `BaseRetriever` (`src/core/interfaces.py`), so it replaces `FakeRetriever` in `run_demo.py` with no other change:

```python
from src.retrieval import build_retriever

agent = AcademicAgent(retriever=build_retriever(), llm=..., prompt_builder=PromptBuilder())
```

`search(query, top_k)` returns a list of `src.core.models.RetrievedChunk`:

| Field | Value |
|---|---|
| `text`, `source` | the chunk and its file name |
| `page` | 1-based page number, or `None` for DOCX/TXT files |
| `score` | dense cosine between the question and the chunk (0 to 1). The agent's `min_score` is compared with this. |
| `extra` | `chunk_id`, `category`, `rank`, `mode`, `embedding_model`, `dot`, `query_norm`, `chunk_norm`, `dense_cosine`, `chroma_distance`, `sparse_cosine`, `sparse_terms`, `dense_rank`, `sparse_rank`, `rrf_score`, `fusion_formula` |

The hybrid ranking decides which chunks are returned; the score shows how close each one is in meaning. An empty query returns an empty list. After each search, `retriever.last_result.explain()` prints the full maths trace of that search.

The richer `retrieve()` method is used by the demos and the evaluation:

```python
retriever = build_retriever()                       # build it once and reuse it

result = retriever.retrieve(question, k=4)          # optional: category="examinations", mode="dense"
result.to_context()        # numbered passages with "Source: file (page n, category c)" for the prompt
result.chunks              # list of RetrievedChunk: .text .source .page .category .citation .scores
result.max_dense_cosine    # best semantic match; refuse to answer when this is too low
result.explain()           # full maths trace (print it in the "Observation" step)
retriever.list_categories()  # lets the agent decide to narrow a search
```

## Run in Google Colab

```python
!pip install -q chromadb pypdf python-docx google-genai
!git clone https://github.com/esala714/RAG---NSBM.git /content/project
!unzip -q -o member1_data.zip -d /content/project    # gives data/raw/... and data/processed/chunks.jsonl
%cd /content/project

# Optional: keep the database between Colab sessions by storing it in Drive
# from google.colab import drive; drive.mount('/content/drive')
# import os; os.environ["CHROMA_DIR"] = "/content/drive/MyDrive/nsbm_vector_db"

!python scripts/build_index.py --reset
!python scripts/search_demo.py "What is the minimum attendance required to sit an exam?"
!python -m pytest -q tests
```

The first run downloads the MiniLM model (about 80 MB). After that it works offline.

To use Gemini embeddings instead, set the key first:
`os.environ["GEMINI_API_KEY"] = userdata.get('geminiKey'); os.environ["EMBEDDING_BACKEND"] = "gemini"`.
They go into their own collection (`nsbm_admin_gemini`), so the two models never mix.

## Run locally (venv)

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/build_index.py --reset
python scripts/search_demo.py            # interactive
pytest -q
```

## Settings (add these to the team's `env.example`)

```
EMBEDDING_BACKEND=minilm      # minilm | gemini
CHROMA_DIR=vector_db
RETRIEVAL_MODE=hybrid         # hybrid | dense | sparse  (compare these in the evaluation)
GEMINI_API_KEY=
```

Add `vector_db/` to `.gitignore`, because the database is rebuilt from the documents.

## The maths, as printed for every retrieved chunk (example values)

```
#1  NSBM academic appeals policy.pdf, page 2  [academic_policies]
    dense : q.d = 0.6123   ||q|| = 1.0000   ||d|| = 1.0000
            cos = q.d / (||q|| x ||d||) = 0.6123   [Chroma distance 0.3877 -> 1 - dist = 0.6123]
    sparse: TF-IDF cos = 0.2736   shared terms (q_i x d_i): appeal:0.081, result:0.067, ...
    fusion: dense rank 1, sparse rank 1  ->  RRF = 1/(60+1) + 1/(60+1) = 0.03279
```

* **Dense:** the cosine is recomputed by hand from the vectors that Chroma returns. It is then checked against Chroma's own distance, where 1 − distance should equal the cosine.
* **Sparse:** `tf = 1 + log(count)`, `idf = log((1+N)/(1+df)) + 1`, and the vectors are L2-normalised. The cosine equals the sum of the `q_i x d_i` products of the shared words.
* **Fusion:** Reciprocal Rank Fusion with k = 60 uses ranks only, so the two score scales never have to be compared directly.

## Design points for the viva

* **Team interface:** `HybridRetriever` inherits `BaseRetriever`, so the agent depends on the abstraction, not on Chroma. `FakeRetriever` (for testing the agent) and `HybridRetriever` (the real search) can replace each other without changing the agent.
* **Abstract base classes:** `BaseEmbedder` and `BaseVectorStore` define what the retriever needs, not how it is done.
* **Strategy pattern:** MiniLM and Gemini are interchangeable embedders.
* **Dependency injection:** the store receives its embedder (and, optionally, a Chroma client), and the retriever receives its store. `factory.py` is the only place that chooses the concrete classes. The tests inject a fake embedder.
* **Idempotent upsert:** chunk ids are hashes of source, page, index and text, so re-indexing never creates duplicates.
* **Single source of truth:** only Chroma is persisted. The TF-IDF index is rebuilt from Chroma when the chunk count changes.
