import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from src.core.interfaces import BaseLLMClient, BaseRetriever
from src.core.models import AgentStep, RAGResponse, RetrievedChunk
from src.rag.llm_client import LLMUnavailableError
from src.rag.prompt_builder import PromptBuilder


class AcademicAgent:
    """Agentic RAG for NSBM academic administration:
    Observation -> Reasoning -> Action -> Persistence."""

    def __init__(self, retriever: BaseRetriever, llm: BaseLLMClient,
                 prompt_builder: PromptBuilder, top_k: int = 4,
                 min_score: float = 0.35, max_retries: int = 2,
                 answer_retries: int = 1,
                 trace_dir: str = "logs/traces"):
        # Dependency Injection: the agent RECEIVES its tools, it doesn't create them
        self.retriever = retriever
        self.llm = llm
        self.prompts = prompt_builder
        self.top_k = top_k
        self.min_score = min_score
        self.max_retries = max_retries
        self.answer_retries = answer_retries
        self.trace_dir = Path(trace_dir)

    def run(self, question: str) -> RAGResponse:
        trace: list[AgentStep] = []

        # 1. OBSERVATION
        trace.append(AgentStep("observation", f"User asked: {question}"))

        # 2. REASONING: plan the search queries
        queries = self._plan(question, trace)
        trace.append(AgentStep("reasoning", f"Planned search queries: {queries}"))

        # 3. ACTION: retrieve evidence for each query (with retries)
        evidence = self._gather_evidence(queries, trace)

        # 4. OBSERVATION: do we have enough evidence?
        if not evidence:
            trace.append(AgentStep("observation", "No chunk passed the score threshold. Refusing to answer."))
            response = RAGResponse(question, self.prompts.NOT_FOUND, [], trace, answerable=False)
            return self._persist(response)

        trace.append(AgentStep("observation",
                               f"Using {len(evidence)} chunks, best score {evidence[0].score:.3f}"))

        # 5. ACTION: generate a grounded answer
        prompt = self.prompts.build_answer_prompt(question, evidence)
        try:
            answer = self.llm.generate(prompt).strip()
        except LLMUnavailableError as error:
            trace.append(AgentStep("observation", f"Gemini unavailable: {error}"))
            response = RAGResponse(question, str(error), evidence, trace, answerable=False)
            return self._persist(response)
        trace.append(AgentStep("action", "Generated answer with Gemini"))

        answerable = self.prompts.NOT_FOUND not in answer

        # 6. OBSERVATION + REASONING: the chunks were similar, but did they hold the answer?
        if not answerable and self.answer_retries > 0:
            retry = self._retry_with_new_queries(question, queries, trace)
            if retry is not None:
                evidence, answer = retry
                answerable = self.prompts.NOT_FOUND not in answer

        response = RAGResponse(question, answer, evidence, trace, answerable)

        # 7. PERSISTENCE
        return self._persist(response)

    # ---------- helper methods ----------

    def _plan(self, question: str, trace: list[AgentStep]) -> list[str]:
        try:
            raw = self.llm.generate(self.prompts.build_planning_prompt(question))
        except LLMUnavailableError:
            trace.append(AgentStep("reasoning", "Gemini busy while planning; searching the original question."))
            return [question]
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            queries = json.loads(raw)
            if isinstance(queries, list) and queries:
                return [str(q) for q in queries[:3]]
        except json.JSONDecodeError:
            pass
        return [question]   # fallback: just search the original question

    def _gather_evidence(self, queries: list[str], trace: list[AgentStep]) -> list[RetrievedChunk]:
        best: dict[tuple, RetrievedChunk] = {}
        for query in queries:
            for chunk in self._search_with_retry(query, trace):
                key = (chunk.source, chunk.page, chunk.text[:60])   # remove duplicates
                if key not in best or chunk.score > best[key].score:
                    best[key] = chunk
        return sorted(best.values(), key=lambda c: c.score, reverse=True)[: self.top_k]

    def _search_with_retry(self, query: str, trace: list[AgentStep]) -> list[RetrievedChunk]:
        for attempt in range(self.max_retries + 1):
            chunks = self.retriever.search(query, self.top_k)
            top = max((c.score for c in chunks), default=0.0)
            trace.append(AgentStep("action",
                                   f"search('{query}') -> {len(chunks)} chunks, top score {top:.3f}"))
            good = [c for c in chunks if c.score >= self.min_score]
            if good:
                return good
            if attempt < self.max_retries:
                trace.append(AgentStep("reasoning",
                                       f"Top score {top:.3f} < threshold {self.min_score}. Rewriting query."))
                try:
                    query = self.llm.generate(self.prompts.build_rewrite_prompt(query)).strip()
                except LLMUnavailableError:
                    trace.append(AgentStep("reasoning", "Gemini busy; cannot rewrite the query."))
                    break
        return []
    

    def _retry_with_new_queries(self, question: str, queries: list[str],
                                trace: list[AgentStep]):
        """Similar chunks are not always useful chunks. If Gemini found no answer,
        rewrite every query once in formal wording, search again and answer again."""
        trace.append(AgentStep("observation",
                               "Gemini found no answer in the chunks, although they passed the threshold."))
        trace.append(AgentStep("reasoning",
                               "Similar is not the same as useful: rewriting the queries in formal wording."))
        try:
            new_queries = [self.llm.generate(self.prompts.build_rewrite_prompt(q)).strip()
                           for q in queries]
        except LLMUnavailableError:
            trace.append(AgentStep("reasoning", "Gemini busy; cannot rewrite the queries."))
            return None
        new_queries = [q for q in new_queries if q]
        trace.append(AgentStep("reasoning", f"New search queries: {new_queries}"))

        evidence = self._gather_evidence(new_queries, trace)
        if not evidence:
            return None
        trace.append(AgentStep("observation",
                               f"Using {len(evidence)} new chunks, best score {evidence[0].score:.3f}"))
        try:
            answer = self.llm.generate(self.prompts.build_answer_prompt(question, evidence)).strip()
        except LLMUnavailableError as error:
            trace.append(AgentStep("observation", f"Gemini unavailable: {error}"))
            return None
        trace.append(AgentStep("action", "Generated a second answer with Gemini"))
        return evidence, answer
    

    def _persist(self, response: RAGResponse) -> RAGResponse:
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        filename = self.trace_dir / f"trace_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
        response.trace.append(AgentStep("persistence", f"Saved trace to {filename}"))
        filename.write_text(json.dumps(asdict(response), indent=2), encoding="utf-8")
        return response