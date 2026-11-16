from src.core.interfaces import BaseRetriever
from src.core.models import RetrievedChunk


class FakeRetriever(BaseRetriever):
    """Returns fixed, made-up chunks so the Agent can be tested without Chroma."""

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                "SAMPLE: Students must have at least 80% attendance in a module "
                "to be eligible to sit its final examination.",
                "exam_regulations_SAMPLE.pdf", 4, 0.81),
            RetrievedChunk(
                "SAMPLE: A student who misses an examination with a valid medical "
                "reason may apply for a repeat attempt within 14 days.",
                "exam_regulations_SAMPLE.pdf", 7, 0.66),
        ][:top_k]


class EmptyRetriever(BaseRetriever):
    """Returns weak matches only, to test the retry and 'not found' path."""

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        return [RetrievedChunk("Unrelated text about the canteen.", "misc.pdf", 1, 0.12)]