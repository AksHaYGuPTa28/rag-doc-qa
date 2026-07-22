import json
from collections.abc import Iterator

from openai import OpenAI

from app.llm.base import ChatLLM


class OpenAILLM(ChatLLM):
    def __init__(self, api_key: str, model: str):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def stream_chat(self, messages: list[dict]) -> Iterator[str]:
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            temperature=0.2,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def generate_structured(self, messages: list[dict], schema: dict, schema_name: str) -> dict:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": schema_name,
                        "description": f"Return a {schema_name} object.",
                        "parameters": schema,
                    },
                }
            ],
            tool_choice={"type": "function", "function": {"name": schema_name}},
        )
        tool_call = response.choices[0].message.tool_calls[0]
        return json.loads(tool_call.function.arguments)
