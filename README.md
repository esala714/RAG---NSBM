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
   OBSERVE  → did the search's best score reach the threshold?
              no  → REASON: rewrite the query and search again (max 2 times)
   ACT      → ask Gemini for an answer using only the retrieved chunks
   OBSERVE  → did Gemini find the answer?
              no  → REASON: rewrite all queries, search and answer once more
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
  tests/                 Offline pytest tests, and test retrievers for the agent
  run_demo.py            Command-line demo of the agent
  requirements.txt       Python packages
  env.example            Template for the .env settings file
```

---

## Setup

Requires **Python 3.11 or newer** (tested on 3.13).

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

**Windows note:** if Python crashes with *"Windows fatal exception: access violation"*
when using Chroma, install the latest **Microsoft Visual C++ Redistributable (x64)**
from Microsoft's official download page, restart the computer, and try again.

## Running the demo

```
python run_demo.py                                       (default question)
python run_demo.py "How much is the repeat exam fee?"    (any question)
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
| Extraction | Reads PDF files page by page with `pypdf`, DOCX files (paragraphs and tables) with `python-docx`, and TXT files as UTF-8. Page numbers start at 1 |
| Cleaning | Turns ligature and maths characters into plain letters (`ﬀ` → `ff`, `𝐺𝑃𝐴` → `GPA`), repairs ligatures written as glyph names (`/f_irst` → `first`), removes null characters, turns tabs into spaces, collapses repeated spaces and keeps at most one blank line between paragraphs |
| Chatbot text | Removes citation marks (`[1, 2]`) and follow-up offers (`If you'd like, let me know: ...`) that an AI chatbot left in 8 of the documents (see below) |
| Repair | Joins words that the PDF text layer split in two (`exa mination` → `examination`), using the vocabulary of the same document (see below) |
| Title page | A cover page with only a title is not chunked on its own; a short title is put in front of every chunk of the document as a heading (see below) |
| Chunking | Splits each page into windows of 180 words with a 40-word overlap, so a sentence cut at the end of one chunk appears whole at the start of the next. Chunking restarts on every page, so a chunk never spans two pages. The heading counts toward the 180 words |
| Labelling | Adds metadata to every chunk: source file, title, heading, category, file type, page number, chunk index, word count and a SHA-256 hash of the file |
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
              "heading": null, "category": "academic_policies", "file_type": "pdf", "page_number": 1,
              "chunk_index": 1, "word_count": 180, "doc_hash": "a418c6f4...74a7"}}
```

| Field | Why it is there |
| --- | --- |
| `source`, `page_number` | The citation shown in the answer, e.g. *NSBM academic appeals policy.pdf, page 1* |
| `heading` | The short title of the document's cover page, also put in front of the chunk text (`null` if the document has none) |
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
[OK] NSBM academic appeals policy.pdf: 5 chunks (2 chatbot offers removed, 12 citation marks removed)
[OK] NSBM Assessment Guideline.pdf: 6 chunks (1 chatbot offer removed, 16 citation marks removed)
[OK] Examination Regulations_Instructions To Candidates.pdf: 12 chunks (13 split words repaired)
[OK] Revision to NSBM EC Guidelines to students from Aug2023.pdf: 6 chunks
[OK] NSBM Degree Programmes_Exam By Laws _ Version 1 (17.2 onwards) _Students.pdf: 6 chunks (2 split words repaired)
[OK] NSBM_Lecturers_Qualifications_Updated_2026.pdf: 5 chunks
[OK] NSBM student handbook 7.pdf: 79 chunks (7 split words repaired)
...
Processed 148 chunks → data/processed/chunks.jsonl
```

| Category | Chunks |
| --- | --- |
| `academic_policies` | 90 |
| `examinations` | 43 |
| `registration` | 6 |
| `staff` | 5 |
| `academic_calendar` | 2 |
| `attendance` | 2 |
| **Total** | **148** (18 files, 148 words per chunk on average) |

The processed file is committed as `data/processed/chunks.jsonl`, so Member 2 can build the
vector store without re-running the extraction.

An AI-written GPA summary that was in the first version of `data/raw/` has been removed: the
official Exam By-Laws (section 3, page 6) already give the GPA formula and the grading scale, so
answers about GPA now cite the university's own document.

**Repairing PDF text.** The text layer of some PDFs is damaged in ways a reader never sees:

| Problem in the extracted text | Example | Fix |
| --- | --- | --- |
| Ligature or maths characters | `eﬀective`, `𝐺𝑃𝐴 = ∑ ...` | Unicode NFKC normalisation |
| Ligature written as its glyph name | `/f_irst`, `in/f_luence` (student handbook) | `/f_` followed by a letter becomes `f` |
| A word split by a space | `exa mination`, `Ca ndidates` (exam regulations) | `SplitWordRepairer` |

`SplitWordRepairer` joins two neighbouring words only when (1) the joined word appears intact
elsewhere in the same document, (2) one of the two pieces is rare there (at most twice), because
real words repeat and broken pieces do not, and (3) neither piece is a common short word, so
"a way" can never become "away". It repaired 22 words in 3 documents, and the number of repairs is printed for every file.
Before these fixes "first" appeared 26 times in the chunks and "field" and "influence" not at all;
now they appear 45, 13 and 16 times, so a search for them finds the student handbook.

**Text written by an AI chatbot.** Eight documents in `data/raw/` are not official NSBM
documents but summaries of NSBM rules that an AI chatbot wrote: the academic appeals policy,
academic calendar, Assessment Guideline, attendance policy, course registration guideline,
"NSBM Green University enforces structured examination regulations", student leave policy and
student registration policy. Next to the facts, they keep two things the chatbot wrote for the
person it was talking to:

| Chatbot text | Example | Why it is removed |
| --- | --- | --- |
| Citation marks (66) | `... at least 50% (combining CA and exam marks). [1, 2]` | They point to web pages that are not in the document, and they look exactly like the agent's own citations, where `[1]` means the first retrieved chunk |
| Follow-up offers (10) | `If you want me to look up something more specific, let me know: • Your Faculty ...` | They name faculties, programmes and intakes, so they match many questions without answering any of them |

`ChatbotTextRemover` deletes every mark, and every offer from its first line to the end of its
paragraph, which holds the list of questions for the user. An offer is found by how it starts
(`If you`, `To give you`, ...) together with a promise in its first 40 words (`let me know`,
`I can help`), so ordinary sentences such as "If you miss an examination, submit an EC form" are
kept. All 10 offers are found, nothing in the official documents is touched, and
`chunks.jsonl` goes from 152 to 148 chunks.
Five of the 14 answerable evaluation questions (q03, q09, q10, q11, q14) are answered from these
documents, so their answers rest on a chatbot's summary rather than on an official NSBM text.

**Title pages.** Three PDFs start with a page that holds only a title, e.g. page 1 of the EC
guidelines: *EXTENUATING CIRCUMSTANCES GUIDELINES FOR EXAMINATIONS* (5 words). As a chunk of its
own it has no answer in it, but it names the topic, so it ranked first for "Is poor internet connectivity
accepted as an extenuating circumstance?" (q13), above page 3, which has the answer but never
uses the word "extenuating". `TitlePageHeading` now treats the leading pages of at most 20 words as the
title page: they are never chunked on their own, and a title of at most 10 words is put in front of
every chunk of that document (`EXTENUATING CIRCUMSTANCES GUIDELINES FOR EXAMINATIONS: ...`).
The chunker keeps room for it, so no chunk is longer than 180 words.

The longer cover pages of the student handbook (11 words, *MBS/MBA STUDENT HANDBOOK MINISTRY OF
HIGHER EDUCATION AND UGC APPROVED PROGRAMMES*) and the Exam By-Laws (17 words, with the intake
year) are dropped without becoming a heading. Measured with the offline evaluation (TF-IDF, k = 4,
before the chatbot text was removed):

| Variant | Chunks | MRR | Context Relevance | Context Recall | Top score, unanswerable |
| --- | --- | --- | --- | --- | --- |
| Before (title pages are chunks) | 155 | 0.857 | 0.500 | 0.841 | 0.191 |
| **Short titles as headings (used)** | 152 | 0.857 | 0.500 | 0.857 | 0.191 |
| Every title page as a heading | 157 | 0.821 | 0.536 | 0.871 | 0.226 |
| Short titles as headings, 150-word chunks | 184 | 0.786 | 0.464 | 0.810 | 0.210 |
| Short titles as headings, 120-word chunks | 223 | 0.857 | 0.446 | 0.813 | 0.266 |

Putting the handbook's 11-word cover text on all of its chunks brings more matching words into the
context, but it adds the same words to all 83 handbook chunks, pushes a handbook chunk above the answer for q09
and raises the score of the unanswerable parking question (0.191 → 0.226), which makes refusing
harder. Smaller chunks do not help either: they cut the answering passages into more pieces, so
less of each passage is retrieved. The text about poor internet in q13 is at word 78 of its
180-word chunk, inside what MiniLM reads, so the chunk size was not the reason it ranked low.

**Running it**

```
pip install -r requirements.txt
python -m src.ingestion.document_processor --input data/raw --output data/processed/chunks.jsonl
python scripts/build_index.py --chunks data/processed/chunks.jsonl --reset    (hand the chunks to Member 2)
python -m pytest -q tests/test_document_processor.py tests/test_ingestion_classes.py   (22 tests, offline)
```

**Design**

```
                    build_processor()   (composition root: the only place that picks the classes)
                           │ injects
                           ▼
 DocumentProcessor ──► BaseExtractor (ABC) ── PdfExtractor │ DocxExtractor │ TxtExtractor
                   ──► TextCleaner           (characters and whitespace, one method per rule)
                   ──► ChatbotTextRemover    (optional; citation marks and follow-up offers)
                   ──► SplitWordRepairer     (optional; uses the whole document's vocabulary)
                   ──► TitlePageHeading      (optional; title page → heading of every chunk)
                   ──► WordChunker           (180-word windows, 40-word overlap)
                   ──► KeywordCategoryClassifier
```

| File | Class | Role |
| --- | --- | --- |
| `src/ingestion/extractors.py` | `BaseExtractor` (ABC), `PdfExtractor`, `DocxExtractor`, `TxtExtractor` | One class per file type (Strategy pattern) |
| `src/ingestion/text_cleaning.py` | `TextCleaner`, `SplitWordRepairer` | Clean and repair the extracted text |
| `src/ingestion/chatbot_text.py` | `ChatbotTextRemover` | Removes what an AI chatbot wrote for its user |
| `src/ingestion/headings.py` | `TitlePageHeading` | Finds the title page and turns a short title into a heading |
| `src/ingestion/chunking.py` | `WordChunker` | Sliding-window chunking |
| `src/ingestion/categories.py` | `KeywordCategoryClassifier` | Category from the file name |
| `src/ingestion/document_processor.py` | `DocumentProcessor`, `build_processor()` | Runs the pipeline and writes `chunks.jsonl` |

- **Interfaces and Dependency Injection:** `DocumentProcessor` receives its extractors, cleaner,
  chatbot-text remover, repairer, heading finder, chunker and classifier through its constructor and only knows the `BaseExtractor`
  interface, so a new format (e.g. OCR for scanned PDFs) is one new class. The tests inject a
  `FakeExtractor` and a 3-word chunker to check the pipeline without any real document.
- **Same interface as before:** the functions `process_file()`, `clean_text()`, `split_into_chunks()`
  and the `SUPPORTED` set still exist (they call the classes), so Member 2's `scripts/build_index.py`
  works unchanged. Before the cleaning fixes were added, the class version produced a
  `chunks.jsonl` identical byte for byte to the function version.
- **Page-level chunking:** chunks never cross a page boundary, so every citation points to one exact page.
- **Safe sliding window:** the window always moves forward by at least one word, so a wrong setting
  (overlap ≥ chunk size) can never cause an endless loop.
- **Fault tolerance:** each file is processed inside its own `try/except`. A damaged file is reported
  as `[SKIP] <file>: <error>` and the remaining files are still processed.
- **Shared data contract:** the output fields match Member 2's `Chunk.from_any()` (`page_number` is
  read as `page`, and the nested `metadata` is read automatically), so no conversion step is needed.

**Known limitations**
- Scanned PDFs (pictures of text) produce no text, because no OCR is used.
- DOCX and TXT files have no page numbers (`page_number` is `null`), so their citations show the file name only.
- The split-word repair is deliberately cautious: a broken word whose joined form never appears
  intact in the same document (e.g. `Addit ional`) is left as it is.
- Categories come from file names, so a badly named file is labelled `general`.
- The chatbot-text removal knows only the offer wordings found in these documents; an offer that
  starts or promises differently would be kept. The facts in the chatbot-written documents are
  kept as they are and are not checked against official NSBM sources.
- A title page is recognised by its length only: a cover page of more than 20 words, or a title in
  the middle of a page, is not found.
- `pypdf` reads a two-column table row by row, so the two columns end up mixed in one chunk (page 3
  of the EC guidelines mixes "Long-term health conditions" with "Technical issues"). That makes
  the chunk match either topic less well.
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

The `AcademicAgent` (`src/agent/academic_agent.py`) answers a question in an explainable agentic loop:

| Phase | What it does |
| --- | --- |
| Observation | Reads the question; checks whether each search reached `min_score`; checks whether Gemini could answer |
| Reasoning | Plans 1–3 search queries in the formal wording that NSBM documents use; rewrites a weak query; after a "not found" answer, rewrites all queries once more |
| Action | Searches the vector store and asks Gemini for an answer from the retrieved chunks only |
| Persistence | Saves the full trace (question, chunks, scores, steps, answer) to `logs/traces/` as JSON |

**Design**
- **Interfaces (ABCs):** `BaseRetriever` and `BaseLLMClient` in `src/core/interfaces.py`.
  Any retriever or LLM that follows them can be used without changing the agent.
- **Dependency Injection:** the agent receives its retriever, LLM client and prompt builder in
  `run_demo.py` (the composition root). `tests/fake_retriever.py` provides `FakeRetriever` and
  `EmptyRetriever` for testing without the database.
- **How `min_score` is used:** it decides whether a search succeeded (its best dense score is at
  least `min_score`). If it did, all its chunks go to the prompt in the hybrid retriever's order,
  so chunks found by exact words (TF-IDF) are not dropped for a low dense score. Chunks from several
  searches are merged round robin (the best chunk of each search first), up to `max_context_chunks`.

**Hallucination control**
- Gemini runs at temperature 0.
- The prompt allows answers **only** from the numbered context and requires citations.
- If no search reaches `min_score`, the agent refuses without asking Gemini for an answer.
- If a search passes but the chunks do not hold the answer, the prompt makes Gemini reply
  "not found" (this caught the canteen and cricket questions, which score above 0.35).

**Error handling**
- If Gemini is busy (HTTP 503), the client retries with increasing waits, then raises a clear
  error that the agent records in the trace instead of crashing.

**Settings** (in `AcademicAgent(...)`)

| Setting | Default | Meaning |
| --- | --- | --- |
| `top_k` | 4 | chunks requested per search |
| `min_score` | 0.35 | dense score a search needs to count as successful (kept after the end-to-end comparison with 0.54) |
| `max_retries` | 2 | rewrites of a weak query |
| `answer_retries` | 1 | extra search round after a "not found" answer |
| `max_context_chunks` | 8 | most chunks sent to Gemini |

**What the agent adds over one search:** in the hybrid retrieval run, q06, q07 and q14 miss the
answering chunk and q13 scores only 0.245, yet the agent answers all four correctly, because it
plans several formal-wording queries and retries.

**Known limitations**
- **Names** (e.g. a lecturer's name) get very low dense scores, so a search can be rejected even
  when TF-IDF finds the right chunk (q12).
- **Answers can be general** when the key detail is elsewhere (q11 misses "3 simple passes").
- **More context, longer answers:** up to 8 chunks can bring in related but unasked facts (still cited).
- **Future work:** a hybrid (dense + sparse) confidence score for accepting a search, tool routing
  for name lookups, and a re-ranker.

---

## User interface (Member 4)

*To be completed by Member 4: how to run the Streamlit app and what it shows.*

---

## Evaluation

The evaluation module (`src/evaluation/`) checks the system against a hand-checked set of questions.

**Ground truth** (`src/evaluation/ground_truth.json`): 17 questions
- **14 answerable questions**, at least one for every document category. Each one is a
  Question / Answer / Context triple: the question, a reference answer, and the passage that
  answers it (`context`, copied word for word from the document). It also names the documents
  that hold the answer (`expected_sources`), the terms a relevant chunk must contain
  (`relevant_terms`) and the keywords a correct answer must contain (`expected_keywords`).
  Tests confirm that every context passage and every term really appear in the source documents,
  so a low score is the system's fault, not the dataset's.
- **3 unanswerable questions** (canteen menu, a cricket result, parking fees): the agent should refuse.

**Metrics** (`src/evaluation/metrics.py`)

| Metric | What it measures | How it is calculated |
| --- | --- | --- |
| Hit@k | Did retrieval find the right document? | 1 if any of the top *k* chunks comes from an expected source, else 0 |
| MRR | How high was the right document ranked? | Mean of 1 / rank of the first chunk from an expected source |
| Chunk Hit@k | Did retrieval find the chunk with the answer? | 1 if any of the top *k* chunks holds at least 90% of the content words of the ground-truth context passage, else 0 |
| Chunk MRR | How high was the chunk with the answer ranked? | Mean of 1 / rank of the first chunk that holds the passage |
| Context Relevance | How much of the context is on topic? | Chunks containing a relevant term / chunks retrieved |
| Context Recall | Was the passage that answers the question retrieved? | Words of the ground-truth context passage found in the retrieved chunks / words of the passage |
| Groundedness (Faithfulness) | Is every claim in the answer backed by the context? | Answer sentences with at least 60% of their content words in the retrieved chunks / sentences checked |
| Answer Relevance | Does the answer contain what was asked for? | Expected keywords found in the answer / expected keywords |
| Refusal accuracy | Does the agent refuse when it should? | Unanswerable questions refused / unanswerable questions |

```math
\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{rank}_q}, \qquad
\text{Groundedness} = \frac{\#\{\, s \in \text{answer} : \text{support}(s) \ge 0.6 \,\}}{\#\{\, s \in \text{answer} \,\}}, \qquad
\text{support}(s) = \frac{|\text{words}(s) \cap \text{words}(\text{context})|}{|\text{words}(s)|}, \qquad
\text{Context Recall} = \frac{|\text{words}(\text{passage}) \cap \text{words}(\text{retrieved})|}{|\text{words}(\text{passage})|}
```

Words are compared with Member 2's tokenizer (lower case, stop words removed, light stemming), so
"examinations" in an answer matches "examination" in a chunk. Citation markers such as [1] are
removed first, and fragments shorter than 3 content words are not checked.

Hit@k and MRR are document-level: a chunk from the right document counts even when it is a title
page or another section. Chunk Hit@k and Chunk MRR are chunk-level and ask whether the answer itself
was retrieved. The 90% line separates the two cleanly: every question has a chunk that holds its
whole passage (100%), and no other chunk holds more than 86% of any passage.

**Results: offline TF-IDF baseline** (`python -m src.evaluation.evaluate`, k = 4,
148 chunks of up to 180 words from 18 documents, with the PDF text repairs, title pages as headings
and the chatbot text removed)

| Metric | Score | Before the chatbot text was removed (152 chunks) | Before the title-page change (155 chunks) | First data version (74 chunks of up to 700 words, 19 documents) |
| --- | --- | --- | --- | --- |
| Hit@4 | 12 / 14 = 0.857 | 12 / 14 = 0.857 | 12 / 14 = 0.857 | 13 / 14 = 0.929 |
| MRR | 0.821 | 0.857 | 0.857 | 0.756 |
| Chunk Hit@4 | 11 / 14 = 0.786 | 11 / 14 = 0.786 | 10 / 14 = 0.714 | not measured |
| Chunk MRR | 0.607 | 0.607 | 0.577 | not measured |
| Context Relevance | 0.482 | 0.500 | 0.500 | 0.464 |
| Context Recall | 0.857 | 0.857 | 0.841 | not measured |

- The right document was in the top 4 for 12 of the 14 questions. The misses are q05 (repeat exam
  fee), where the word "examination" ranks the longer examination-rules documents above the one-page
  fee table, and q14 (intake months), where chunks of the student handbook rank above the academic
  calendar. Matching exact words is not enough on its own, which is why Member 2 adds dense
  (meaning-based) search.
- Smaller chunks put the right chunk higher when it is found (MRR 0.756 → 0.857) and bring less
  off-topic text into the context (Context Relevance 0.464 → 0.500), but TF-IDF alone misses one
  more question. The main reason for 180-word chunks is dense search (see Member 1), which this
  offline run does not use.
- On average 86% of the words of the answering passage are in the 4 retrieved chunks. Context
  Recall also catches what Hit@4 misses: q11 counts as a hit because the registration policy is an
  accepted source, but the passage with the A/L entry requirements is in the course registration
  guideline, so its recall is only 0.40.
- The chunk-level scores show the gap that document-level scores hide: Chunk MRR (0.607) is well
  below MRR (0.821). q11 counts as a document hit, but no retrieved chunk holds the A/L requirements;
  q04, q06, q09 and q12 have the answering chunk only at rank 4, 4, 2 and 2.
- With the heading on every chunk of the EC guidelines, q04 (deadline for an EC claim) now gets
  the whole answering passage (Context Recall 0.77 → 1.00). Before the change, q13 counted as a
  document hit at rank 1 because of the title page, while the chunk with the answer was only at
  rank 3; now it is at rank 1 (Chunk Hit@4 10 → 11, Chunk MRR 0.577 → 0.607).
- Removing the chatbot text moves the answering chunk of q02 from rank 2 to 1 and that of q09 from
  rank 1 to 2, so Chunk MRR stays at 0.607. MRR falls to 0.821 only because of q09, where a chunk
  of the postgraduate assessment criteria now scores 0.185 and the Assessment Guideline 0.180.
- TF-IDF scores cannot separate answerable from unanswerable questions: the parking question scores
  0.191, above four answerable questions (0.187, 0.185, 0.174 and 0.138). The agent's refusal threshold
  (`min_score`) should therefore be chosen from the dense cosine scores of the `--retriever hybrid`
  run, between the two "top score" values that the report prints.

**Results: full system** (hybrid retrieval with MiniLM and the agent with Gemini, k = 4, run by
Member 2 on the 155 chunks from before the title-page change and the chatbot-text removal, so its
TF-IDF column is that version's; they will be run again on the 148 chunks;
the threshold study is in [`README_member2.md`](README_member2.md))

| Retrieval metric | Hybrid (`--retriever hybrid`) | TF-IDF baseline |
| --- | --- | --- |
| Hit@4 | 14 / 14 = 1.000 | 12 / 14 = 0.857 |
| MRR | 0.881 | 0.857 |
| Context Relevance | 0.500 | 0.500 |
| Context Recall | 0.906 | 0.841 |

| Answer metric (`--with-agent`, `min_score` 0.35) | Score |
| --- | --- |
| Answerable questions answered | 9 / 14 |
| Groundedness (Faithfulness) | 1.000 |
| Answer Relevance | 0.643 |
| Refusal accuracy (3 unanswerable questions) | 1.000 |
| False refusal rate | 0.357 |

- Hybrid search finds the right document for all 14 questions, including q05 and q14, which TF-IDF
  misses because the question and the document use different words. More of each answering passage
  reaches the prompt (Context Recall 0.841 → 0.906); for q11 it rises from 0.40 to 1.00, because the
  course registration guideline now comes back.
- Every checked sentence of every answer was supported by the retrieved chunks (Groundedness 1.000),
  and the agent refused all three questions the documents cannot answer, so the metrics found no
  unsupported answer.
- Answer Relevance (0.643) equals the share of answerable questions answered (9 / 14): every answer
  given contained all of its expected keywords, and each refusal scores 0.
- The weakness is false refusals: q04, q09, q11, q12 and q13 were refused although retrieval finds
  their document (Hit@4 = 1). The loss happens in the agent step, where Gemini plans the search
  queries and only chunks scoring at least `min_score` reach the prompt, so that step is where the
  next improvement belongs.
- Member 2's end-to-end check compared `min_score` 0.35 and 0.54: the stricter value refused more
  answerable questions without refusing more unanswerable ones, so the agent keeps 0.35. Single runs
  vary by a question or two, because Gemini's planned queries change between runs, and 14 + 3
  questions give indications, not precise rates.

**Running it**

```
python -m src.evaluation.evaluate                                   (offline TF-IDF baseline, no API key)
python -m src.evaluation.evaluate --retriever hybrid                (Member 2's Chroma store: run scripts/build_index.py first)
python -m src.evaluation.evaluate --retriever hybrid --with-agent   (answers from Gemini: needs GEMINI_API_KEY and GEMINI_MODEL in .env)
python -m pytest -q tests/test_evaluation.py                        (18 tests, run offline)
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
- **Free-tier pacing:** the free Gemini tier allows 15 requests per minute, and the agent makes
  several per question (search plan, query rewrites, answer, answer check), so an unpaced run stops
  with `429 RESOURCE_EXHAUSTED` after a few questions. `PacedLLM` (`src/evaluation/pacing.py`)
  wraps the Gemini client: it leaves 5 seconds between calls (`--requests-per-minute 12`, the
  default) and, if a 429 still comes, waits as long as the error asks and tries again. A full run
  of 17 questions therefore takes several minutes, and prints each question as it is answered.

**Known limitations**
- Groundedness and answer relevance compare words, not meaning: a correct paraphrase can score
  lower, and a sentence can reuse words from the context and still be wrong.
- 17 questions is a small dataset, although it covers every document category.

---

## Demo video

*Link to the unlisted YouTube video: to be added.*