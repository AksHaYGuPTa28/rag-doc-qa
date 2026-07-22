from collections.abc import Iterator

from anthropic import Anthropic

from app.llm.base import ChatLLM


class ClaudeLLM(ChatLLM):
    def __init__(self, api_key: str, model: str):
        self.client = Anthropic(api_key=api_key)
        self.model = model

    @staticmethod
    def _split_system(messages: list[dict]) -> tuple[str, list[dict]]:
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        rest = [m for m in messages if m["role"] != "system"]
        return "\n".join(system_parts), rest

    def stream_chat(self, messages: list[dict]) -> Iterator[str]:
        system, rest = self._split_system(messages)
        with self.client.messages.stream(
            model=self.model,
            max_tokens=1024,
            system=system,
            messages=rest,
        ) as stream:
            yield from stream.text_stream

    def generate_structured(self, messages: list[dict], schema: dict, schema_name: str) -> dict:
        system, rest = self._split_system(messages)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system,
            messages=rest,
            tools=[
                {
                    "name": schema_name,
                    "description": f"Return a {schema_name} object.",
                    "input_schema": schema,
                }
            ],
            tool_choice={"type": "tool", "name": schema_name},
        )
        for block in response.content:
            if block.type == "tool_use":
                return block.input
        return {}
