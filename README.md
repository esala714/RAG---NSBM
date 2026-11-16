# Explainable Agentic RAG System – NSBM Academic Administration

An AI assistant that answers students' and staff's questions about NSBM Green University
academic rules and procedures (exam regulations, attendance, registration, repeats).
It answers **only from the university's documents**, shows **which document and page**
each fact came from, and records **how** it reached each answer.

Built for **DS205.3 – Data Science with Python** (NSBM Green University).

---

## Team and responsibilities

| Member | Responsibility | Main code |
| --- | --- | --- |
| Member 1 | Document processing: PDF extraction, cleaning, chunking, metadata | `src/ingestion/` |
| Member 2 | Embeddings and vector database (Chroma), similarity search | `src/retrieval/` |
| Member 3 | Agentic RAG pipeline: planning, retrieval loop, grounded answers, traces | `src/core/`, `src/rag/`, `src/agent/` |
| Member 4 | Streamlit user interface and integration | `Frontend/` |

---

## How it works

```
PDF documents
   │  (Member 1) extract, clean, chunk
   ▼
Vector store (Chroma)
   │  (Member 2) embeddings + similarity search
   ▼
Agent (Member 3)
   OBSERVE  → read the question
   REASON   → plan 1–3 search queries
   ACT      → search the vector store
   OBSERVE  → check similarity scores against a threshold
              too low → REASON: rewrite the query and search again
   ACT      → ask Gemini for an answer using only the retrieved chunks
   PERSIST  → save the full trace as JSON
   ▼
Streamlit app (Member 4) shows the answer, sources, pages and scores
```

---

## Project structure

```
RAG---NSBM/
  Frontend/              Streamlit user interface
  src/
    core/                Shared data models and interfaces (ABCs)
    rag/                 Gemini client and prompt builder
    agent/               The agentic loop
    retrieval/           Embeddings, Chroma vector store and hybrid search
   scripts/               Build the vector store and run search demos
   tests/                 Test retrievers used before the vector store is connected
  run_demo.py            Command-line demo of the agent
  requirements.txt       Python packages
  env.example            Template for the .env settings file
```

---

## Setup

Requires **Python 3.10 or newer**.

1. Clone the repository and open the folder.
2. Create and activate a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate          (Windows)
   source venv/bin/activate       (Mac/Linux)
   ```
3. Install the packages:
   ```
   pip install -r requirements.txt
   ```
4. Copy `env.example`, rename the copy to `.env`, and add your own Gemini API key
   (free from Google AI Studio):
   ```
   GEMINI_API_KEY=your-key-here
   GEMINI_MODEL=gemini-3.5-flash-lite
   ```
   The `.env` file is never committed to GitHub.

---

## Running the demo

```
python run_demo.py
```

The demo prints:
- **Agent trace**: each Observation, Reasoning, Action and Persistence step
- **Retrieved chunks**: the document text used, with source, page and similarity score
- **Final answer**: a grounded answer citing its sources as [1], [2]

Each run is also saved as a JSON file in `logs/traces/`.

---

## Document processing (Member 1)

*To be completed by Member 1: how PDFs are extracted, cleaned and chunked, and what metadata is attached.*

---

## Embeddings and retrieval (Member 2)

The retrieval layer (`src/retrieval/`) turns Member 1's chunks into vectors, stores them in a
persistent Chroma database, and returns the most relevant chunks for a question **together with
the maths behind every score**, so each retrieval decision can be checked by hand.

| Step | What it does |
| --- | --- |
| Embedding | Each chunk becomes a 384-number vector with `all-MiniLM-L6-v2` (runs locally through Chroma, no API key) |
| Storage | Chroma `PersistentClient` saves the vectors, text and metadata (source, category, page) in `vector_db/`, so nothing is lost between sessions. Re-indexing never creates duplicates (each chunk id is a hash of its source, page and text) |
| Dense search | Finds chunks with the same **meaning** (cosine similarity in Chroma) |
| Sparse search | Finds chunks with the same **exact words**, such as "MBA" or "C+" (TF-IDF, written from scratch with NumPy) |
| Fusion | Combines the two ranked lists with Reciprocal Rank Fusion (RRF) |

**How the similarity scores are calculated**

Dense score: cosine similarity between the question vector *q* and a chunk vector *d*:

```math
\cos(q, d) = \frac{q \cdot d}{\lVert q \rVert \, \lVert d \rVert}, \qquad q \cdot d = \sum_{i=1}^{384} q_i d_i, \qquad \lVert q \rVert = \sqrt{\sum_{i=1}^{384} q_i^2}
```

- MiniLM vectors are already normalised (‖q‖ = ‖d‖ = 1), so here the cosine equals the dot product.
- The cosine is recomputed by hand from the vectors Chroma returns and checked against Chroma's own
  cosine distance (cos = 1 − distance).

Sparse score: TF-IDF vectors, normalised to length 1, compared with the same cosine formula.
Rare words weigh more: tf = 1 + log(count), idf = log((1 + N) / (1 + df)) + 1.

Fusion: only the ranks are combined, because the two score scales are not comparable (k = 60):

```math
\mathrm{RRF}(d) = \frac{1}{60 + \mathrm{rank}_{\mathrm{dense}}(d)} + \frac{1}{60 + \mathrm{rank}_{\mathrm{sparse}}(d)}
```

**How the scores are logged**

`result.explain()` prints every intermediate value. Real output for
*"How late can a candidate enter the examination hall?"*:

```
#1  Examination Regulations_Instructions To Candidates.pdf, page 2  [Exam]
    dense : q.d = 0.6589   ||q|| = 1.0000   ||d|| = 1.0000
            cos = q.d / (||q|| x ||d||) = 0.6589   [Chroma distance 0.3411 -> 1 - dist = 0.6589]
    sparse: TF-IDF cos = 0.2736   shared terms (q_i x d_i): hall:0.081, enter:0.067, candidate:0.065
    fusion: dense rank 1, sparse rank 1  ->  RRF = 1/(60+1) + 1/(60+1) = 0.03279
```

The word "late" never appears in the documents, yet the dense search still finds the rule on
admission times. This is why both searches are used.

**Running it**

```
python scripts/build_index.py --raw data/raw --reset      (build the vector store once)
python scripts/search_demo.py "How late can a candidate enter the examination hall?"
python scripts/search_demo.py "First class GPA" --mode dense   (modes: hybrid, dense, sparse)
python -m pytest -q tests                                   (14 tests, run offline)
```

**Design**
- **Interfaces (ABCs):** `BaseEmbedder` and `BaseVectorStore`. The embedding model can be swapped
  (MiniLM or Gemini) without changing the store or the search code (Strategy pattern).
- **Dependency Injection:** the store receives its embedder and the retriever receives its store.
  `factory.py` is the only place that chooses them. The tests inject a fake embedder, so they run
  offline in seconds.

Full details: [`README_member2.md`](README_member2.md).

---

## Agentic RAG pipeline (Member 3)

The `AcademicAgent` (`src/agent/academic_agent.py`) runs an explainable agentic loop:

| Phase | What it does |
| --- | --- |
| Observation | Reads the question; later checks whether retrieval scores are good enough |
| Reasoning | Uses Gemini to split complex questions into 1–3 search queries; rewrites weak queries |
| Action | Searches the vector store and asks Gemini for an answer from the retrieved chunks only |
| Persistence | Saves the full trace (question, chunks, scores, steps, answer) to `logs/traces/` |

**Design**
- **Interfaces (ABCs):** `BaseRetriever` and `BaseLLMClient` in `src/core/interfaces.py`.
  Any retriever or LLM that follows them can be used without changing the agent.
- **Dependency Injection:** the agent receives its retriever, LLM client and prompt
  builder when it is created (see `run_demo.py`), so they can be swapped easily.

**Hallucination control**
- Gemini runs at temperature 0.
- The prompt allows answers **only** from the numbered context and requires citations.
- Chunks below a similarity threshold (`min_score`, default 0.35) are not used.
- If no chunk passes the threshold, the agent replies that the information was not found.

**Error handling**
- If Gemini is busy (HTTP 503), the client retries with increasing waits before
  reporting a clear error, and the agent records it in the trace instead of crashing.

**Settings** (in `AcademicAgent(...)`): `top_k` (chunks used, default 4),
`min_score` (similarity threshold, default 0.35), `max_retries` (query rewrites, default 2).

---

## User interface (Member 4)

*To be completed by Member 4: how to run the Streamlit app and what it shows.*

---

## Evaluation

*To be completed: the ground-truth dataset and how Context Relevance, Groundedness
(Faithfulness) and Answer Relevance are calculated, with results.*

---

## Demo video

*Link to the unlisted YouTube video: to be added.*