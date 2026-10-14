from openai import AsyncOpenAI


class OpenAIEmbeddingProvider:
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        dimensions: int,
        batch_size: int = 20,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size 必须大于 0")
        self._client = client
        self._model = model
        self._dimensions = dimensions
        self._batch_size = batch_size

    @property
    def model_name(self) -> str:
        return self._model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            response = await self._client.embeddings.create(
                model=self._model,
                input=texts[start : start + self._batch_size],
                dimensions=self._dimensions,
                encoding_format="float",
            )
            ordered_items = sorted(response.data, key=lambda item: item.index)
            vectors.extend(item.embedding for item in ordered_items)
        return vectors

    async def embed_one(self, text: str) -> list[float]:
        return (await self.embed([text]))[0]
