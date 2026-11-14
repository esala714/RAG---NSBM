import os
from dotenv import load_dotenv

from src.agent.academic_agent import AcademicAgent
from src.rag.llm_client import GeminiClient
from src.rag.prompt_builder import PromptBuilder
from tests.fake_retriever import FakeRetriever   # later: Member 2's real class

load_dotenv()

# Composition root: all objects are created here and injected
agent = AcademicAgent(
    retriever=FakeRetriever(),
    llm=GeminiClient(os.getenv("GEMINI_API_KEY"), os.getenv("GEMINI_MODEL")),
    prompt_builder=PromptBuilder(),
)

question = ("What attendance do I need to sit the final exam, "
            "and what can I do if I miss the exam because I was sick?")
result = agent.run(question)

print("\n=== AGENT TRACE ===")
for step in result.trace:
    print(f"[{step.phase.upper():12}] {step.detail}")

print("\n=== RETRIEVED CHUNKS (before generation) ===")
for i, c in enumerate(result.sources, 1):
    print(f"[{i}] score={c.score:.3f} | {c.source} p.{c.page}")
    print(f"    {c.text[:120]}")

print("\n=== FINAL ANSWER ===")
print(result.answer)