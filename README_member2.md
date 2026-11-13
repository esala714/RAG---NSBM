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
| `src/retrieval/retriever.py` | `HybridRetriever`: dense + sparse + Reciprocal Rank Fusion, with the score trace |
| `src/retrieval/factory.py` | The composition root, `build_retriever()`, which reads settings from the environment |
| `scripts/build_index.py` | Indexes Member 1's chunks (JSONL), or `data/raw` for a quick test |
| `scripts/search_demo.py` | Prints retrieved chunks and their maths (use this for the video) |
| `tests/test_retrieval.py` | 14 pytest tests that run offline with a fake embedder |

## Interface with the team

**Input from Member 1** is a list of dicts (or a JSONL file with one dict per line):

```json
{"text": "2.3 Candidates shall be admitted ...", "source": "Examination Regulations_Instructions To Candidates.pdf", "category": "Exam", "page": 2, "chunk_index": 1}
```

`category` is the folder name and `page` starts at 1. Common alternative names are also accepted: `page_content`/`content`, `document_name`/`file_name`, `page_number`, and a nested `metadata` dict.

```python
store.add_chunks(chunks)   # in Python, or: python scripts/build_index.py --chunks data/processed/chunks.jsonl
```

**Output to Member 3 (RAG and agent):**

```python
from src.retrieval import build_retriever
retriever = build_retriever()                       # build it once and reuse it

result = retriever.retrieve(question, k=4)          # optional: category="GPA", mode="dense"
result.to_context()        # numbered passages with "Source: file (page n)" for the prompt
result.chunks              # list of RetrievedChunk: .text .source .page .category .scores
result.max_dense_cosine    # best semantic match; refuse to answer when this is too low
result.explain()           # full maths trace (print it in the "Observation" step)
retriever.list_categories()  # lets the agent decide to narrow a search
```

## Run in Google Colab

```python
!pip install -q chromadb pypdf python-docx google-genai
!unzip -q -o member2_retrieval.zip -d /content/project
!unzip -q -o data_ZIP.zip -d /content/project        # gives /content/project/data/raw/...
%cd /content/project

# Optional: keep the database between Colab sessions by storing it in Drive
# from google.colab import drive; drive.mount('/content/drive')
# import os; os.environ["CHROMA_DIR"] = "/content/drive/MyDrive/nsbm_vector_db"

!python scripts/build_index.py --raw data/raw --reset      # until Member 1's chunks.jsonl is ready
!python scripts/search_demo.py "How late can a candidate enter the examination hall?"
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
python scripts/build_index.py --raw data/raw --reset
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
#1  Examination Regulations_Instructions To Candidates.pdf, page 2  [Exam]
    dense : q.d = 0.6123   ||q|| = 1.0000   ||d|| = 1.0000
            cos = q.d / (||q|| x ||d||) = 0.6123   [Chroma distance 0.3877 -> 1 - dist = 0.6123]
    sparse: TF-IDF cos = 0.2736   shared terms (q_i x d_i): hall:0.081, enter:0.067, ...
    fusion: dense rank 1, sparse rank 1  ->  RRF = 1/(60+1) + 1/(60+1) = 0.03279
```

* **Dense:** the cosine is recomputed by hand from the vectors that Chroma returns. It is then checked against Chroma's own distance, where 1 − distance should equal the cosine.
* **Sparse:** `tf = 1 + log(count)`, `idf = log((1+N)/(1+df)) + 1`, and the vectors are L2-normalised. The cosine equals the sum of the `q_i x d_i` products of the shared words.
* **Fusion:** Reciprocal Rank Fusion with k = 60 uses ranks only, so the two score scales never have to be compared directly.

## Design points for the viva

* **Abstract base classes:** `BaseEmbedder` and `BaseVectorStore` define what the retriever needs, not how it is done.
* **Strategy pattern:** MiniLM and Gemini are interchangeable embedders.
* **Dependency injection:** the store receives its embedder (and, optionally, a Chroma client), and the retriever receives its store. `factory.py` is the only place that chooses the concrete classes. The tests inject a fake embedder.
* **Idempotent upsert:** chunk ids are hashes of source, page, index and text, so re-indexing never creates duplicates.
* **Single source of truth:** only Chroma is persisted. The TF-IDF index is rebuilt from Chroma when the chunk count changes.
