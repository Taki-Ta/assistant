from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from zhiyin.agent.context.budget import ContextBudget
from zhiyin.agent.context.builder import ContextBuilder
from zhiyin.agent.preferences import PreferenceService
from zhiyin.agent.providers.openai import OpenAIProvider
from zhiyin.agent.repository import ConversationRepository
from zhiyin.agent.runtime import AgentContext, ToolDependencies
from zhiyin.agent.services import ConversationService, SessionService
from zhiyin.api.auth import CurrentUserId
from zhiyin.config import config
from zhiyin.db.database import get_conversation_db, get_db, session_factory
from zhiyin.db.repositories import PostgresChunkStore
from zhiyin.indexing.index_service import IndexService
from zhiyin.indexing.protocols import EmbeddingProvider
from zhiyin.indexing.search_service import SearchService
from zhiyin.indexing.vector_store import PostgresVectorStore

DatabaseSession = Annotated[AsyncSession, Depends(get_db)]
ConversationDatabaseSession = Annotated[
    AsyncSession,
    Depends(get_conversation_db),
]


async def get_conversation_service(
    db: ConversationDatabaseSession,
) -> AsyncIterator[ConversationService]:
    budget = ContextBudget.configured()
    async with OpenAIProvider(budget=budget) as provider:
        yield ConversationService(
            _db=db,
            _repository=ConversationRepository(db),
            _provider=provider,
            _context_builder=ContextBuilder(summarizer=provider, budget=budget),
        )


ConversationServiceDependency = Annotated[
    ConversationService,
    Depends(get_conversation_service),
]


def get_embedding_provider(request: Request) -> EmbeddingProvider:
    return request.app.state.embedding_provider


EmbeddingProviderDependency = Annotated[
    EmbeddingProvider,
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


def get_agent_context(
    owner_id: CurrentUserId,
) -> AgentContext:
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
        preference_service=PreferenceService(session_factory),
    )


ToolDependenciesDependency = Annotated[
    ToolDependencies,
    Depends(get_tool_dependencies),
]


IndexServiceDependency = Annotated[IndexService, Depends(get_index_service)]


def get_session_service(db: ConversationDatabaseSession) -> SessionService:
    return SessionService(db, ConversationRepository(db))


SessionServiceDependency = Annotated[SessionService, Depends(get_session_service)]
