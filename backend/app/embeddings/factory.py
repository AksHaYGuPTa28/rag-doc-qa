from app.config import Settings
from app.embeddings.base import Embedder
from app.embeddings.local_embedder import LocalEmbedder
from app.embeddings.openai_embedder import OpenAIEmbedder


def get_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider == "local":
        return LocalEmbedder(settings.local_embedding_model)
    if settings.embedding_provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set but EMBEDDING_PROVIDER=openai")
        return OpenAIEmbedder(api_key=settings.openai_api_key, model=settings.openai_embedding_model)
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {settings.embedding_provider}")
