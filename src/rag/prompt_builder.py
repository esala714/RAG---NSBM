from src.core.models import RetrievedChunk


class PromptBuilder:
    NOT_FOUND = ("I could not find this information in the NSBM academic documents. "
                 "Please contact your faculty office.")

    def build_answer_prompt(self, question: str, chunks: list[RetrievedChunk]) -> str:
        context = "\n\n".join(
            f"[{i}] (source: {c.source}, page {c.page})\n{c.text}"
            for i, c in enumerate(chunks, start=1)
        )
        return f"""You are an academic administration assistant for NSBM Green University.
You help students and staff with academic rules and procedures.
Rules:
- Answer ONLY using the numbered context below. Do not use outside knowledge.
- After each fact, cite its source number, like [1] or [2].
- If the context does not contain the answer, reply exactly: "{self.NOT_FOUND}"

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

    def build_planning_prompt(self, question: str) -> str:
        return f"""Break this university academic administration question into 1 to 3 short
search queries that would find the needed rules in NSBM academic documents
(exam regulations, attendance rules, registration, repeats, student handbook).
Return ONLY a JSON list of strings, for example: ["minimum attendance requirement", "exam eligibility"]

QUESTION: {question}"""

    def build_rewrite_prompt(self, query: str) -> str:
        return f"""The search query below found no relevant NSBM academic documents.
Rewrite it using different, more formal university regulation wording.
Return ONLY the new query, nothing else.

QUERY: {query}"""



    







