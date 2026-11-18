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
  data/
    raw/                 The 18 NSBM source documents (PDF), one folder per topic
    processed/           chunks.jsonl: Member 1's output, Member 2's input
  src/
    ingestion/           PDF/DOCX extraction, cleaning, chunking and metadata
    core/                Shared data models and interfaces (ABCs)
    rag/                 Gemini client and prompt builder
    agent/               The agentic loop
    retrieval/           Embeddings, Chroma vector store and hybrid search
    evaluation/          Ground-truth questions, metrics and the evaluation runner
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

The ingestion layer (`src/ingestion/document_processor.py`) turns the raw NSBM documents in
`data/raw/` into small, clean, labelled pieces of text (**chunks**) that Member 2 can embed.
Every chunk keeps **which file and which page** it came from, so the agent's final answer can
cite its source exactly.

| Step | What it does |
| --- | --- |
| Extraction | Reads PDF files page by page with `pypdf`, DOCX files with `python-docx`, and TXT files as UTF-8. Page numbers start at 1 |
| Cleaning | Removes null characters, turns tabs and other control whitespace into single spaces, collapses repeated spaces and keeps at most one blank line between paragraphs |
| Chunking | Splits each page into windows of 180 words with a 40-word overlap, so a sentence cut at the end of one chunk appears whole at the start of the next. Chunking restarts on every page, so a chunk never spans two pages |
| Labelling | Adds metadata to every chunk: source file, title, category, file type, page number, chunk index, word count and a SHA-256 hash of the file |
| Export | Writes one chunk per line to `data/processed/chunks.jsonl`, the input for `scripts/build_index.py --chunks` |

**How the chunks are calculated**

With chunk size *S* = 180 words and overlap *O* = 40 words, each window starts *S − O* = 140
words after the previous one:

```math
\text{start}_{k+1} = \text{start}_k + (S - O), \qquad \text{chunk}_k = \text{words}[\,\text{start}_k : \text{start}_k + S\,]
```

So a page with *n* words gives (an empty page gives 0 chunks):

```math
\text{chunks}(n) = \begin{cases} 1 & 1 \le n \le S \\ \left\lceil \dfrac{n - O}{S - O} \right\rceil & n > S \end{cases}
```

Check: the 8-word test sentence with *S* = 4 and *O* = 2 gives ⌈(8 − 2) / 2⌉ = 3 chunks, exactly
what `test_chunk_overlap` expects.

**Why 180 words:** Member 2's embedding model (MiniLM) reads only the first 256 word-pieces of a
chunk, about 192 words, and ignores the rest. With the first setting (700 words, 100 overlap)
46 of the 74 chunks were longer than that, so dense search never saw most of their text.
180-word chunks fit inside the limit, and the 40-word overlap still keeps sentences whole.
Both values can be changed with `--chunk-size` and `--overlap`.

**What one chunk looks like**

```json
{"id": "NSBM academic appeals policy-1-1",
 "text": "Academic Appeals Policy NSBM Green University handles academic appeals ...",
 "metadata": {"source": "NSBM academic appeals policy.pdf", "doc_title": "NSBM academic appeals policy",
              "category": "academic_policies", "file_type": "pdf", "page_number": 1,
              "chunk_index": 1, "word_count": 180, "doc_hash": "a418c6f4...74a7"}}
```

| Field | Why it is there |
| --- | --- |
| `source`, `page_number` | The citation shown in the answer, e.g. *NSBM academic appeals policy.pdf, page 1* |
| `category` | Lets the retriever filter by topic (examinations, registration, attendance, ...) |
| `chunk_index` | Position of the chunk inside its page |
| `word_count` | Quick check for empty or very short chunks |
| `doc_hash` | SHA-256 of the whole file: it changes when a document is replaced, so re-processing can be detected |

The category is chosen from keywords in the file name, and the first matching rule wins:
*calendar* → `academic_calendar`, *registration* → `registration`, *exam / assessment / repeat* →
`examinations`, *attendance / leave* → `attendance`, *lecturer / staff* → `staff`,
*policy / handbook / guideline* → `academic_policies`. Files that match nothing are labelled `general`.

**Real output** for the 18 NSBM documents in `data/raw/`:

```
[OK] NSBM academic appeals policy.pdf: 5 chunks
[OK] Examination Regulations_Instructions To Candidates.pdf: 12 chunks
[OK] NSBM Degree Programmes_Exam By Laws _ Version 1 (17.2 onwards) _Students.pdf: 7 chunks
[OK] NSBM_Lecturers_Qualifications_Updated_2026.pdf: 5 chunks
[OK] NSBM student handbook 7.pdf: 80 chunks
...
Processed 155 chunks → data/processed/chunks.jsonl
```

| Category | Chunks |
| --- | --- |
| `academic_policies` | 92 |
| `examinations` | 45 |
| `registration` | 6 |
| `staff` | 5 |
| `attendance` | 4 |
| `academic_calendar` | 3 |
| **Total** | **155** (18 files, 145 words per chunk on average) |

The processed file is committed as `data/processed/chunks.jsonl`, so Member 2 can build the
vector store without re-running the extraction.

An AI-written GPA summary that was in the first version of `data/raw/` has been removed: the
official Exam By-Laws (section 3, page 6) already give the GPA formula and the grading scale, so
answers about GPA now cite the university's own document.

**Running it**

```
pip install -r requirements-member1.txt
python -m src.ingestion.document_processor --input data/raw --output data/processed/chunks.jsonl
python scripts/build_index.py --chunks data/processed/chunks.jsonl --reset    (hand the chunks to Member 2)
python -m pytest -q tests/test_document_processor.py                           (7 tests, run offline)
```

**Design**
- **One function per step:** `extract_pages` → `clean_text` → `split_into_chunks` → `process_file`.
  Each step is tested on its own, and the tests need no real NSBM documents
  (`test_process_file_metadata` creates a temporary text file).
- **Page-level chunking:** chunks never cross a page boundary, so every citation points to one exact page.
- **Safe sliding window:** the window always moves forward by at least one word, so a wrong setting
  (overlap ≥ chunk size) can never cause an endless loop.
- **Fault tolerance:** each file is processed inside its own `try/except`. A damaged file is reported
  as `[SKIP] <file>: <error>` and the remaining files are still processed.
- **Shared data contract:** the output fields match Member 2's `Chunk.from_any()` (`page_number` is
  read as `page`, and the nested `metadata` is read automatically), so no conversion step is needed.

**Known limitations**
- Scanned PDFs (pictures of text) produce no text, because no OCR is used.
- DOCX and TXT files have no page numbers (`page_number` is `null`); Member 2's loader treats them as page 1.
- Categories come from file names, so a badly named file is labelled `general`.
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

The evaluation module (`src/evaluation/`) checks the system against a hand-checked set of questions.

**Ground truth** (`src/evaluation/ground_truth.json`): 17 questions
- **14 answerable questions**, at least one for every document category. Each one names the
  document that holds the answer (`expected_sources`), the terms a relevant chunk must contain
  (`relevant_terms`), the keywords a correct answer must contain (`expected_keywords`) and a
  reference answer. A test confirms that every term really
  appears in its source document, so a low score is the system's fault, not the dataset's.
- **3 unanswerable questions** (canteen menu, a cricket result, parking fees): the agent should refuse.

**Metrics** (`src/evaluation/metrics.py`)

| Metric | What it measures | How it is calculated |
| --- | --- | --- |
| Hit@k | Did retrieval find the right document? | 1 if any of the top *k* chunks comes from an expected source, else 0 |
| MRR | How high was the right document ranked? | Mean of 1 / rank of the first chunk from an expected source |
| Context Relevance | How much of the context is on topic? | Chunks containing a relevant term / chunks retrieved |
| Groundedness (Faithfulness) | Is every claim in the answer backed by the context? | Answer sentences with at least 60% of their content words in the retrieved chunks / sentences checked |
| Answer Relevance | Does the answer contain what was asked for? | Expected keywords found in the answer / expected keywords |
| Refusal accuracy | Does the agent refuse when it should? | Unanswerable questions refused / unanswerable questions |

```math
\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{rank}_q}, \qquad
\text{Groundedness} = \frac{\#\{\, s \in \text{answer} : \text{support}(s) \ge 0.6 \,\}}{\#\{\, s \in \text{answer} \,\}}, \qquad
\text{support}(s) = \frac{|\text{words}(s) \cap \text{words}(\text{context})|}{|\text{words}(s)|}
```

Words are compared with Member 2's tokenizer (lower case, stop words removed, light stemming), so
"examinations" in an answer matches "examination" in a chunk. Citation markers such as [1] are
removed first, and fragments shorter than 3 content words are not checked.

**Results: offline TF-IDF baseline** (`python -m src.evaluation.evaluate`, k = 4,
155 chunks of up to 180 words from 18 documents)

| Metric | Score | First data version (74 chunks of up to 700 words, 19 documents) |
| --- | --- | --- |
| Hit@4 | 12 / 14 = 0.857 | 13 / 14 = 0.929 |
| MRR | 0.857 | 0.756 |
| Context Relevance | 0.518 | 0.464 |

- The right document was in the top 4 for 12 of the 14 questions. The misses are q05 (repeat exam
  fee), where the word "examination" ranks the longer examination-rules documents above the one-page
  fee table, and q14 (intake months), where chunks of the student handbook rank above the academic
  calendar. Matching exact words is not enough on its own, which is why Member 2 adds dense
  (meaning-based) search.
- Smaller chunks put the right chunk higher when it is found (MRR 0.756 → 0.857) and bring less
  off-topic text into the context (Context Relevance 0.464 → 0.518), but TF-IDF alone misses one
  more question. The main reason for 180-word chunks is dense search (see Member 1), which this
  offline run does not use.
- TF-IDF scores cannot separate answerable from unanswerable questions: the parking question scores
  0.175, above three answerable questions (0.172, 0.154 and 0.134). The agent's refusal threshold
  (`min_score`) should therefore be chosen from the dense cosine scores of the `--retriever hybrid`
  run, between the two "top score" values that the report prints.

*Hybrid retrieval and agent answer results: to be added after running the commands below.*

**Running it**

```
python -m src.evaluation.evaluate                                   (offline TF-IDF baseline, no API key)
python -m src.evaluation.evaluate --retriever hybrid                (Member 2's Chroma store: run scripts/build_index.py first)
python -m src.evaluation.evaluate --retriever hybrid --with-agent   (answers from Gemini: needs GEMINI_API_KEY in .env)
python -m pytest -q tests/test_evaluation.py                        (12 tests, run offline)
```

Each run prints a table with one row per question and saves the full results (every retrieved
chunk, every answer sentence and its support score) as JSON in `logs/evaluation/`.

**Design**
- **Same interface as the agent:** the evaluation only calls `BaseRetriever.search()`, exactly as
  the agent does. `--retriever hybrid` uses Member 2's `HybridRetriever` as it is (its score is the
  dense cosine), and `TfidfRetriever` is an offline baseline over Member 1's chunks.
- **Dependency Injection:** the retriever and the agent are passed in, so the tests use
  `FakeRetriever` and a scripted agent, and run without Chroma or Gemini.
- **Fair averages:** questions where Gemini was unavailable are reported and left out of the
  averages instead of being counted as refusals.

**Known limitations**
- Groundedness and answer relevance compare words, not meaning: a correct paraphrase can score
  lower, and a sentence can reuse words from the context and still be wrong.
- 17 questions is a small dataset, although it covers every document category.

---

## Demo video

*Link to the unlisted YouTube video: to be added.*