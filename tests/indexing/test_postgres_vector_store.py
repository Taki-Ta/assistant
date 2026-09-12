from unittest.mock import AsyncMock, Mock
from uuid import UUID

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import AsyncSession

from interview_ai.db.models import VectorRecord
from interview_ai.indexing.vector_store import PostgresVectorStore


def _session(*, execute_result: object | None = None) -> Mock:
    session = Mock(spec=AsyncSession)
    session.execute = AsyncMock(return_value=execute_result)
    return session


@pytest.mark.asyncio
async def test_postgres_vector_store_upserts_by_chunk_id() -> None:
    session = _session()
    store = PostgresVectorStore(session)
    record = VectorRecord(
        chunk_id=UUID("550e8400-e29b-41d4-a716-446655440001"),
        vector=[0.1, 0.2],
        embedding_model="test-embedding",
    )

    await store.upsert([record])

    statement = session.execute.await_args.args[0]
    sql = str(statement.compile(dialect=postgresql.dialect()))
    assert "INSERT INTO vectors" in sql
    assert "ON CONFLICT (chunk_id) DO UPDATE" in sql


@pytest.mark.asyncio
async def test_postgres_vector_store_search_converts_distance_to_similarity() -> None:
    chunk_id = UUID("550e8400-e29b-41d4-a716-446655440001")
    result = Mock()
    result.all.return_value = [(chunk_id, 0.25)]
    session = _session(execute_result=result)

    matches = await PostgresVectorStore(session).search([0.1, 0.2], limit=3)

    assert matches[0].chunk_id == chunk_id
    assert matches[0].score == pytest.approx(0.75)


@pytest.mark.asyncio
async def test_postgres_vector_store_skips_empty_writes() -> None:
    session = _session()
    store = PostgresVectorStore(session)

    await store.upsert([])
    await store.delete([])

    session.execute.assert_not_awaited()
