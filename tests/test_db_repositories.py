from unittest.mock import AsyncMock, Mock
from uuid import NAMESPACE_URL, uuid5

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from interview_ai.db.models import Chunk
from interview_ai.db.repositories import PostgresChunkStore


def _chunk(document_id):
    return Chunk(
        id=uuid5(NAMESPACE_URL, f"{document_id}-0-hash"),
        document_id=document_id,
        sort_index=0,
        level=1,
        headings=("Python",),
        content="# Python",
        start_line=1,
        end_line=1,
        hash="a" * 64,
    )


@pytest.mark.asyncio
async def test_chunk_store_replaces_document_chunks_without_committing() -> None:
    document_id = uuid5(NAMESPACE_URL, "document")
    chunks = [_chunk(document_id)]
    session = Mock(spec=AsyncSession)
    session.execute = AsyncMock()
    session.flush = AsyncMock()

    await PostgresChunkStore(session).replace(document_id, chunks)

    session.execute.assert_awaited_once()
    session.add_all.assert_called_once_with(chunks)
    session.flush.assert_awaited_once()
    session.commit.assert_not_called()


@pytest.mark.asyncio
async def test_chunk_store_rejects_chunks_from_another_document() -> None:
    document_id = uuid5(NAMESPACE_URL, "document")
    session = Mock(spec=AsyncSession)
    session.execute = AsyncMock()
    session.flush = AsyncMock()

    with pytest.raises(ValueError, match="document_id"):
        await PostgresChunkStore(session).replace(
            document_id,
            [_chunk(uuid5(NAMESPACE_URL, "another-document"))],
        )

    session.execute.assert_not_awaited()
