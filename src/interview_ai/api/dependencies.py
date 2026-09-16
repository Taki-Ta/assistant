from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from interview_ai.agent.providers.openai import OpenAIProvider
from interview_ai.agent.repository import ConversationRepository
from interview_ai.agent.runtime import AgentContext, ToolDependencies
from interview_ai.agent.services import ConversationService
from interview_ai.api.auth import verify_bearer
from interview_ai.config import config
from interview_ai.db.database import get_conversation_db, get_db
from interview_ai.db.repositories import PostgresChunkStore
from interview_ai.indexing.embedding import OpenAIEmbeddingProvider
from interview_ai.indexing.search_service import SearchService
from interview_ai.indexing.service import IndexService
from interview_ai.indexing.vector_store import PostgresVectorStore

DatabaseSession = Annotated[AsyncSession, Depends(get_db)]
ConversationDatabaseSession = Annotated[
    AsyncSession,
    Depends(get_conversation_db),
]


async def get_conversation_service(
    db: ConversationDatabaseSession,
) -> AsyncIterator[ConversationService]:
    async with OpenAIProvider() as provider:
        yield ConversationService(
            _db=db,
            _repository=ConversationRepository(db),
            _provider=provider,
        )


ConversationServiceDependency = Annotated[
    ConversationService,
    Depends(get_conversation_service),
]


def get_embedding_provider(request: Request) -> OpenAIEmbeddingProvider:
    return request.app.state.embedding_provider


EmbeddingProviderDependency = Annotated[
    OpenAIEmbeddingProvider,
    Depends(get_embedding_provider),
]


def get_search_service(
    db: DatabaseSession,
    embedding_provider: EmbeddingProviderDependency,
) -> SearchService:
    return SearchService(
        embedding_provider=embedding_provider,
        vector_store=PostgresVectorStore(db),
        default_limit=config.search_top_k,
        score_threshold=config.search_score_threshold,
    )


def get_index_service(
    db: DatabaseSession,
    embedding_provider: EmbeddingProviderDependency,
) -> IndexService:
    return IndexService(
        embedding_provider=embedding_provider,
        chunk_store=PostgresChunkStore(db),
        vector_store=PostgresVectorStore(db),
    )


Claims = Annotated[dict[str, object], Depends(verify_bearer)]


def get_agent_context(
    claims: Claims,
) -> AgentContext:
    owner_id = claims.get("sub")

    if not isinstance(owner_id, str) or not owner_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token subject is missing",
        )

    return AgentContext(owner_id=owner_id)


AgentContextDependency = Annotated[
    AgentContext,
    Depends(get_agent_context),
]

SearchServiceDependency = Annotated[SearchService, Depends(get_search_service)]


def get_tool_dependencies(
    search_service: SearchServiceDependency,
) -> ToolDependencies:
    return ToolDependencies(
        search_service=search_service,
    )


ToolDependenciesDependency = Annotated[
    ToolDependencies,
    Depends(get_tool_dependencies),
]


IndexServiceDependency = Annotated[IndexService, Depends(get_index_service)]
