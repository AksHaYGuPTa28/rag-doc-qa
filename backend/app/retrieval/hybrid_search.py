from rank_bm25 import BM25Okapi

from app.embeddings.base import Embedder
from app.models.schemas import DocumentChunk, ScoredChunk
from app.retrieval.vector_store import VectorStore

# Reciprocal Rank Fusion constant -- standard default from the RRF paper,
# dampens the influence of any single ranker's top hit.
_RRF_K = 60


class HybridRetriever:
    """Combines dense vector similarity with sparse BM25 keyword matching.

    Each ranker contributes 1 / (RRF_K + rank) per hit; summing across both
    rankers (Reciprocal Rank Fusion) rewards chunks that either ranker likes,
    without needing the two scores to be on comparable scales.
    """

    def __init__(self, vector_store: VectorStore, embedder: Embedder):
        self._vector_store = vector_store
        self._embedder = embedder

    def retrieve(self, query: str, top_k: int, fetch_k: int | None = None) -> list[ScoredChunk]:
        fetch_k = fetch_k or top_k * 4

        query_embedding = self._embedder.embed_query(query)
        vector_hits = self._vector_store.similarity_search(query_embedding, k=fetch_k)
        keyword_hits = self._keyword_search(query, fetch_k)

        fused: dict[str, ScoredChunk] = {}

        for rank, (chunk, score) in enumerate(vector_hits):
            entry = fused.setdefault(chunk.chunk_id, ScoredChunk(chunk=chunk))
            entry.vector_score = score
            entry.fused_score += 1.0 / (_RRF_K + rank + 1)

        for rank, (chunk, score) in enumerate(keyword_hits):
            entry = fused.setdefault(chunk.chunk_id, ScoredChunk(chunk=chunk))
            entry.keyword_score = score
            entry.fused_score += 1.0 / (_RRF_K + rank + 1)

        return sorted(fused.values(), key=lambda c: c.fused_score, reverse=True)[:top_k]

    def _keyword_search(self, query: str, fetch_k: int) -> list[tuple[DocumentChunk, float]]:
        chunks = self._vector_store.all_chunks()
        if not chunks:
            return []

        tokenized_corpus = [c.text.lower().split() for c in chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(query.lower().split())

        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:fetch_k]
        return [(chunks[i], float(scores[i])) for i in ranked_idx]
