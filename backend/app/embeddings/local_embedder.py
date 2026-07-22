from app.embeddings.base import Embedder


class LocalEmbedder(Embedder):
    """Runs entirely on-device via sentence-transformers -- no API key required.

    Lets the app index and hybrid-search documents even when only an
    ANTHROPIC_API_KEY is configured (Claude has no embeddings endpoint).
    """

    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "EMBEDDING_PROVIDER=local requires the 'sentence-transformers' "
                "package. Install it with: pip install sentence-transformers"
            ) from exc
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
