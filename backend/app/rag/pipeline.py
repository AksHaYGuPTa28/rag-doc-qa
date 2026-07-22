import uuid
from collections.abc import Iterator

from app.config import Settings
from app.embeddings.base import Embedder
from app.ingestion.chunker import chunk_pages
from app.ingestion.loader import load_file
from app.llm.base import ChatLLM
from app.models.schemas import AnswerWithCitations, ChatMessage, ScoredChunk, SourceMeta, UploadResponse
from app.rag.prompts import ANSWER_SCHEMA, build_messages, build_sources
from app.retrieval.hybrid_search import HybridRetriever
from app.retrieval.reranker import Reranker
from app.retrieval.vector_store import VectorStore


class RAGPipeline:
    """Orchestrates ingestion and question-answering end to end:
    ingest() -> load/chunk/embed/store; retrieve() -> hybrid search + rerank;
    stream_answer() / answer_structured() -> grounded generation."""

    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        vector_store: VectorStore,
        retriever: HybridRetriever,
        reranker: Reranker,
        llm: ChatLLM,
    ):
        self._settings = settings
        self._embedder = embedder
        self._vector_store = vector_store
        self._retriever = retriever
        self._reranker = reranker
        self._llm = llm

    def ingest(self, file_path: str, source_name: str) -> UploadResponse:
        pages = load_file(file_path)
        document_id = str(uuid.uuid4())
        chunks = chunk_pages(
            pages,
            document_id,
            source_name,
            self._settings.chunk_size,
            self._settings.chunk_overlap,
        )
        if chunks:
            embeddings = self._embedder.embed_documents([c.text for c in chunks])
            self._vector_store.add_chunks(chunks, embeddings)
        return UploadResponse(document_id=document_id, source=source_name, chunks_indexed=len(chunks))

    def retrieve(self, query: str) -> list[ScoredChunk]:
        candidates = self._retriever.retrieve(query, top_k=self._settings.retrieval_top_k)
        return self._reranker.rerank(query, candidates, top_k=self._settings.rerank_top_k)

    def stream_answer(self, query: str, history: list[ChatMessage]) -> tuple[Iterator[str], list[SourceMeta]]:
        scored_chunks = self.retrieve(query)
        sources = build_sources(scored_chunks)
        messages = build_messages(query, scored_chunks, history)
        return self._llm.stream_chat(messages), sources

    def answer_structured(self, query: str, history: list[ChatMessage]) -> AnswerWithCitations:
        """Non-streaming path that uses function/tool calling to force a
        machine-readable {answer, citations, confidence} JSON payload."""
        scored_chunks = self.retrieve(query)
        messages = build_messages(query, scored_chunks, history)
        raw = self._llm.generate_structured(messages, ANSWER_SCHEMA, schema_name="answer_with_citations")
        return AnswerWithCitations.model_validate(raw)

    def list_documents(self) -> list[dict]:
        return self._vector_store.list_documents()
