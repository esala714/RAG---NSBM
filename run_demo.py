import os
import sys
from dotenv import load_dotenv

from src.agent.academic_agent import AcademicAgent
from src.rag.llm_client import GeminiClient
from src.rag.prompt_builder import PromptBuilder
from src.retrieval import build_retriever            # Member 2's real hybrid retriever
# from tests.fake_retriever import FakeRetriever     # for testing without the database

load_dotenv()

# Composition root: all objects are created here and injected
retriever = build_retriever()
agent = AcademicAgent(
    retriever=retriever,
    llm=GeminiClient(os.getenv("GEMINI_API_KEY"), os.getenv("GEMINI_MODEL")),
    prompt_builder=PromptBuilder(),
)

DEFAULT_QUESTION = ("What attendance do I need to sit an exam, "
                    "and on what grounds can I appeal my exam results?")
question = " ".join(sys.argv[1:]) or DEFAULT_QUESTION
result = agent.run(question)

print("\n=== AGENT TRACE ===")
for step in result.trace:
    print(f"[{step.phase.upper():12}] {step.detail}")

print("\n=== RETRIEVED CHUNKS (before generation) ===")
for i, c in enumerate(result.sources, 1):
    print(f"[{i}] score={c.score:.3f} | {c.source} p.{c.page}")
    maths = {key: c.extra.get(key) for key in
             ("dot", "query_norm", "chunk_norm", "sparse_cosine", "rrf_score")}
    print(f"    maths: {maths}")
    print(f"    {c.text[:120]}")

print("\n=== FINAL ANSWER ===")
print(result.answer)