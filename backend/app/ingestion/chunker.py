import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingestion.loader import LoadedPage
from app.models.schemas import DocumentChunk


def chunk_pages(
    pages: list[LoadedPage],
    document_id: str,
    source: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[DocumentChunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[DocumentChunk] = []
    for page in pages:
        for piece in splitter.split_text(page.text):
            if not piece.strip():
                continue
            chunks.append(
                DocumentChunk(
                    chunk_id=str(uuid.uuid4()),
                    document_id=document_id,
                    source=source,
                    page=page.page,
                    text=piece.strip(),
                )
            )
    return chunks
