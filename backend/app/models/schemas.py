from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    source: str
    page: int | None = None
    text: str


class ScoredChunk(BaseModel):
    chunk: DocumentChunk
    vector_score: float = 0.0
    keyword_score: float = 0.0
    fused_score: float = 0.0
    rerank_score: float | None = None


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    query: str
    history: list[ChatMessage] = Field(default_factory=list)


class Citation(BaseModel):
    marker: int
    source: str
    page: int | None = None
    quote: str


class AnswerWithCitations(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: str = Field(description="one of: high, medium, low")


class SourceMeta(BaseModel):
    marker: int
    source: str
    page: int | None = None
    preview: str


class UploadResponse(BaseModel):
    document_id: str
    source: str
    chunks_indexed: int


class DocumentSummary(BaseModel):
    document_id: str
    source: str
    chunk_count: int
