from openai import AsyncOpenAI


class OpenAIEmbeddingProvider:
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        dimensions: int,
    ) -> None:
        self._client = client
        self._model = model
        self._dimensions = dimensions

    @property
    def model_name(self) -> str:
        return self._model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        response = await self._client.embeddings.create(
            model=self._model,
            input=texts,
            dimensions=self._dimensions,
            encoding_format="float",
        )
        ordered_items = sorted(response.data, key=lambda item: item.index)
        return [item.embedding for item in ordered_items]

    async def embed_one(self, text: str) -> list[float]:
        return (await self.embed([text]))[0]
