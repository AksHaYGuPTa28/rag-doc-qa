from functools import lru_cache

from app.config import get_settings
from app.embeddings.factory import get_embedder
from app.llm.factory import get_llm
from app.rag.pipeline import RAGPipeline
from app.retrieval.hybrid_search import HybridRetriever
from app.retrieval.reranker import get_reranker
from app.retrieval.vector_store import VectorStore


@lru_cache
def get_pipeline() -> RAGPipeline:
    """Builds the RAG pipeline once and reuses it across requests. Acts as
    the composition root wiring the pluggable embedder/LLM/reranker."""
    settings = get_settings()
    embedder = get_embedder(settings)
    vector_store = VectorStore(settings.chroma_persist_dir, settings.chroma_collection)
    retriever = HybridRetriever(vector_store, embedder)
    reranker = get_reranker()
    llm = get_llm(settings)
    return RAGPipeline(settings, embedder, vector_store, retriever, reranker, llm)
