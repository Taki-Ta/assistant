from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager

from langchain_openai import OpenAIEmbeddings
from openai import AsyncOpenAI, OpenAI

from zhiyin.config import Settings

from ..protocols import EmbeddingProvider
from .langchain import LangChainEmbeddingProvider
from .openai import OpenAIEmbeddingProvider


@asynccontextmanager
async def create_embedding_provider(
    settings: Settings,
) -> AsyncGenerator[EmbeddingProvider]:
    """Create the configured provider and close its clients on exit."""
    async with AsyncExitStack() as stack:
        client = await stack.enter_async_context(
            AsyncOpenAI(api_key=settings.api_key, base_url=settings.host)
        )
        provider: EmbeddingProvider
        if settings.embedding_provider == "langchain":
            sync_client = stack.enter_context(
                OpenAI(api_key=settings.api_key, base_url=settings.host)
            )
            embeddings = OpenAIEmbeddings(
                model=settings.embedding_model_name,
                openai_api_key=settings.api_key,
                openai_api_base=settings.host,
                dimensions=settings.dimensions,
                chunk_size=settings.embedding_batch_size,
                # OpenAI-compatible services may only accept string inputs.
                check_embedding_ctx_length=False,
                client=sync_client.embeddings,
                async_client=client.embeddings,
            )
            provider = LangChainEmbeddingProvider(embeddings)
        else:
            provider = OpenAIEmbeddingProvider(
                client=client,
                model=settings.embedding_model_name,
                dimensions=settings.dimensions,
                batch_size=settings.embedding_batch_size,
            )
        yield provider
