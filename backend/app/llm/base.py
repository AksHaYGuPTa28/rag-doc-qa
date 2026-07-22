from abc import ABC, abstractmethod
from collections.abc import Iterator


class ChatLLM(ABC):
    """Provider-agnostic chat interface. Implementations wrap OpenAI or Claude."""

    @abstractmethod
    def stream_chat(self, messages: list[dict]) -> Iterator[str]:
        """Yield answer text incrementally, token/segment by token/segment."""

    @abstractmethod
    def generate_structured(self, messages: list[dict], schema: dict, schema_name: str) -> dict:
        """Force the model to return JSON matching `schema` via tool/function calling."""
