from app.config import Settings
from app.llm.base import ChatLLM
from app.llm.claude_llm import ClaudeLLM
from app.llm.gemini_llm import GeminiLLM
from app.llm.openai_llm import OpenAILLM


def get_llm(settings: Settings) -> ChatLLM:
    if settings.llm_provider == "anthropic":
        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set but LLM_PROVIDER=anthropic")
        return ClaudeLLM(api_key=settings.anthropic_api_key, model=settings.anthropic_chat_model)
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set but LLM_PROVIDER=openai")
        return OpenAILLM(api_key=settings.openai_api_key, model=settings.openai_chat_model)
    if settings.llm_provider == "gemini":
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not set but LLM_PROVIDER=gemini")
        return GeminiLLM(api_key=settings.gemini_api_key, model=settings.gemini_chat_model)
    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider}")
