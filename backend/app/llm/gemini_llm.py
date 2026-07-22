import json
from collections.abc import Iterator

from google import genai
from google.genai import types

from app.llm.base import ChatLLM


class GeminiLLM(ChatLLM):
    def __init__(self, api_key: str, model: str):
        self.client = genai.Client(api_key=api_key)
        self.model = model

    @staticmethod
    def _split_system(messages: list[dict]) -> tuple[str, list[dict]]:
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        rest = [m for m in messages if m["role"] != "system"]
        return "\n".join(system_parts), rest

    @staticmethod
    def _to_contents(messages: list[dict]) -> list[dict]:
        # Gemini uses "model" instead of "assistant" for the assistant role.
        return [
            {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
            for m in messages
        ]

    def stream_chat(self, messages: list[dict]) -> Iterator[str]:
        system, rest = self._split_system(messages)
        stream = self.client.models.generate_content_stream(
            model=self.model,
            contents=self._to_contents(rest),
            config=types.GenerateContentConfig(system_instruction=system or None),
        )
        for chunk in stream:
            if chunk.text:
                yield chunk.text

    def generate_structured(self, messages: list[dict], schema: dict, schema_name: str) -> dict:
        system, rest = self._split_system(messages)
        response = self.client.models.generate_content(
            model=self.model,
            contents=self._to_contents(rest),
            config=types.GenerateContentConfig(
                system_instruction=system or None,
                response_mime_type="application/json",
                response_json_schema=schema,
            ),
        )
        return json.loads(response.text)
