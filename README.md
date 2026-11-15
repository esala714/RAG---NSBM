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

*To be completed by Member 2: embedding model, Chroma setup, and how similarity scores
(dot product, vector magnitude, cosine similarity) are calculated and logged.*

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