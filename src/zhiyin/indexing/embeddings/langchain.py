from langchain_openai import OpenAIEmbeddings


class LangChainEmbeddingProvider:
    def __init__(self, client: OpenAIEmbeddings) -> None:
        self.client = client

    @property
    def model_name(self) -> str:
        return self.client.model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        return await self.client.aembed_documents(texts)

    async def embed_one(self, text: str) -> list[float]:
        result = await self.embed([text])
        if len(result) != 1 or not result[0]:
            raise ValueError("单条 Embedding 请求必须返回一个非空向量")
        return result[0]
