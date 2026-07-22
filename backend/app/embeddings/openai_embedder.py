from openai import OpenAI

from app.embeddings.base import Embedder


class OpenAIEmbedder(Embedder):
    def __init__(self, api_key: str, model: str):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in response.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
