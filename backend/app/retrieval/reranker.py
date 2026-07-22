from abc import ABC, abstractmethod

from app.models.schemas import ScoredChunk

_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        ...


class NoOpReranker(Reranker):
    """Falls back to the hybrid-fusion order when no cross-encoder is installed."""

    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        return candidates[:top_k]


class CrossEncoderReranker(Reranker):
    """Rescoring stage: a cross-encoder jointly reads (query, chunk) pairs,
    which is more accurate than the bi-encoder similarity used for the
    initial vector search, at the cost of only being cheap enough to run
    on the small shortlist retrieval already narrowed down."""

    def __init__(self, model_name: str = _CROSS_ENCODER_MODEL):
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        if not candidates:
            return []
        pairs = [(query, c.chunk.text) for c in candidates]
        scores = self._model.predict(pairs)
        for candidate, score in zip(candidates, scores):
            candidate.rerank_score = float(score)
        return sorted(candidates, key=lambda c: c.rerank_score, reverse=True)[:top_k]


def get_reranker() -> Reranker:
    try:
        return CrossEncoderReranker()
    except ImportError:
        return NoOpReranker()
