from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM generation
    llm_provider: str = "openai"  # openai | anthropic | gemini
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    gemini_api_key: str | None = None
    openai_chat_model: str = "gpt-4o-mini"
    anthropic_chat_model: str = "claude-sonnet-5"
    gemini_chat_model: str = "gemini-flash-latest"

    # Embeddings
    embedding_provider: str = "openai"  # openai | local
    openai_embedding_model: str = "text-embedding-3-small"
    local_embedding_model: str = "all-MiniLM-L6-v2"

    # Retrieval tuning
    chunk_size: int = 800
    chunk_overlap: int = 120
    retrieval_top_k: int = 8
    rerank_top_k: int = 4

    # Storage
    chroma_persist_dir: str = "./data/chroma"
    chroma_collection: str = "documents"
    upload_dir: str = "./data/uploads"

    # CORS
    frontend_origin: str = "http://localhost:3000"


@lru_cache
def get_settings() -> Settings:
    return Settings()
