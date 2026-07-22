import chromadb

from app.models.schemas import DocumentChunk


class VectorStore:
    """Thin wrapper around a persistent Chroma collection."""

    def __init__(self, persist_dir: str, collection_name: str):
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(name=collection_name)

    def add_chunks(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "document_id": c.document_id,
                    "source": c.source,
                    "page": c.page if c.page is not None else -1,
                }
                for c in chunks
            ],
        )

    def similarity_search(self, query_embedding: list[float], k: int) -> list[tuple[DocumentChunk, float]]:
        count = self._collection.count()
        if count == 0:
            return []
        result = self._collection.query(query_embeddings=[query_embedding], n_results=min(k, count))

        hits: list[tuple[DocumentChunk, float]] = []
        for cid, doc, meta, dist in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            hits.append((_chunk_from_metadata(cid, doc, meta), 1.0 / (1.0 + dist)))
        return hits

    def all_chunks(self) -> list[DocumentChunk]:
        """Fetch every stored chunk. Used to (re)build the in-memory BM25 index.

        Fine for a demo-scale corpus; a production system would persist the
        BM25 index (or use a search engine with native keyword support)
        instead of rebuilding it from the full collection on every query.
        """
        count = self._collection.count()
        if count == 0:
            return []
        result = self._collection.get(limit=count)
        return [
            _chunk_from_metadata(cid, doc, meta)
            for cid, doc, meta in zip(result["ids"], result["documents"], result["metadatas"])
        ]

    def list_documents(self) -> list[dict]:
        docs: dict[str, dict] = {}
        for chunk in self.all_chunks():
            entry = docs.setdefault(
                chunk.document_id, {"document_id": chunk.document_id, "source": chunk.source, "chunk_count": 0}
            )
            entry["chunk_count"] += 1
        return list(docs.values())


def _chunk_from_metadata(chunk_id: str, text: str, meta: dict) -> DocumentChunk:
    page = meta.get("page")
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=meta["document_id"],
        source=meta["source"],
        page=None if page in (None, -1) else page,
        text=text,
    )
