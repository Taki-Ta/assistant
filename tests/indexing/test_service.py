from unittest.mock import AsyncMock, Mock

import pytest

from interview_ai.indexing.protocols import (
    ChunkStore,
    EmbeddingProvider,
    VectorStore,
)
from interview_ai.indexing.service import IndexService
from interview_ai.ingestion.scanner import split_document
from tests.ingestion.util import make_file


def make_service(vectors):
    embedding_provider = Mock(spec=EmbeddingProvider)
    embedding_provider.model_name = "test-embedding"
    embedding_provider.embed.return_value = vectors
    chunk_store = AsyncMock(spec=ChunkStore)
    vector_store = AsyncMock(spec=VectorStore)
    service = IndexService(embedding_provider, chunk_store, vector_store)
    return service, embedding_provider, chunk_store, vector_store


@pytest.mark.asyncio
async def test_index_persists_chunks_and_vectors(tmp_path):
    document = make_file(
        "# Python\n\n基础内容。\n\n## 异常\n\n异常处理。",
        tmp_path / "python.md",
    )
    chunks = split_document(document, max_chunk_length=20)
    vectors = [[float(index), 1.0] for index in range(len(chunks))]
    service, embedding_provider, chunk_store, vector_store = make_service(vectors)

    count = await service.index(document, chunks)

    assert count == len(chunks)
    embedding_provider.embed.assert_called_once_with(
        [chunk.content for chunk in chunks]
    )
    chunk_store.replace.assert_awaited_once_with(document.id, chunks)
    vector_store.upsert.assert_awaited_once()

    records = vector_store.upsert.call_args.args[0]
    assert len(records) == len(chunks)
    for record, chunk, vector in zip(records, chunks, vectors, strict=True):
        assert record.chunk_id == chunk.id
        assert record.vector == vector
        assert record.embedding_model == "test-embedding"


@pytest.mark.asyncio
async def test_index_with_no_chunks_clears_existing_chunks(tmp_path):
    document = make_file("", tmp_path / "empty.md")
    service, embedding_provider, chunk_store, vector_store = make_service([])

    count = await service.index(document, [])

    assert count == 0
    embedding_provider.embed.assert_not_called()
    chunk_store.replace.assert_awaited_once_with(document.id, [])
    vector_store.upsert.assert_awaited_once_with([])


@pytest.mark.asyncio
async def test_index_rejects_chunk_without_matching_document(tmp_path):
    document = make_file("# Orphan", tmp_path / "orphan.md")
    chunks = split_document(document)
    other_document = make_file("# Other", tmp_path / "other.md")
    service, embedding_provider, chunk_store, vector_store = make_service(
        [[1.0, 0.0]]
    )

    with pytest.raises(ValueError, match="Chunk 找不到对应 Document"):
        await service.index(other_document, chunks)

    embedding_provider.embed.assert_not_called()
    chunk_store.replace.assert_not_awaited()
    vector_store.upsert.assert_not_awaited()


@pytest.mark.asyncio
async def test_index_rejects_embedding_count_mismatch(tmp_path):
    document = make_file("# Python", tmp_path / "python.md")
    chunks = split_document(document)
    service, embedding_provider, chunk_store, vector_store = make_service([])

    with pytest.raises(ValueError, match="向量数量与 Chunk 数量不一致"):
        await service.index(document, chunks)

    embedding_provider.embed.assert_called_once_with(
        [chunk.content for chunk in chunks]
    )
    chunk_store.replace.assert_not_awaited()
    vector_store.upsert.assert_not_awaited()

