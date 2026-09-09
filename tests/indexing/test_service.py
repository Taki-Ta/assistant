from unittest.mock import Mock

import pytest

from interview_ai.indexing.embedding import EmbeddingProvider
from interview_ai.indexing.service import IndexService
from interview_ai.indexing.vector_store import VectorStore
from interview_ai.ingestion.scanner import split_file
from tests.ingestion.util import make_file


def make_service(vectors):
    embedding_provider = Mock(spec=EmbeddingProvider)
    embedding_provider.embed.return_value = vectors
    vector_store = Mock(spec=VectorStore)
    service = IndexService(embedding_provider, vector_store)
    return service, embedding_provider, vector_store


def test_index_builds_vector_records_and_writes_them_once(tmp_path):
    document = make_file(
        "# Python\n\n基础内容。\n\n## 异常\n\n异常处理。",
        tmp_path / "python.md",
    )
    chunks = split_file(document, max_chunk_length=20)
    vectors = [[float(index), 1.0] for index in range(len(chunks))]
    service, embedding_provider, vector_store = make_service(vectors)

    count = service.index([document], chunks)

    assert count == len(chunks)
    embedding_provider.embed.assert_called_once_with(
        [chunk.content for chunk in chunks]
    )
    vector_store.upsert.assert_called_once()

    records = vector_store.upsert.call_args.args[0]
    assert len(records) == len(chunks)
    for record, chunk, vector in zip(records, chunks, vectors, strict=True):
        assert record.chunk_id == chunk.id
        assert record.document_id == document.id
        assert record.vector == vector
        assert record.content == chunk.content
        assert record.metadata.path == document.path
        assert record.metadata.headings == chunk.headings
        assert record.metadata.start_line == chunk.start_line
        assert record.metadata.end_line == chunk.end_line


def test_index_with_no_chunks_skips_embedding_and_storage(tmp_path):
    document = make_file("", tmp_path / "empty.md")
    service, embedding_provider, vector_store = make_service([])

    count = service.index([document], [])

    assert count == 0
    embedding_provider.embed.assert_not_called()
    vector_store.upsert.assert_not_called()


def test_index_rejects_chunk_without_matching_document(tmp_path):
    document = make_file("# Orphan", tmp_path / "orphan.md")
    chunks = split_file(document)
    service, embedding_provider, vector_store = make_service([[1.0, 0.0]])

    with pytest.raises(ValueError, match="Chunk 找不到对应 Document"):
        service.index([], chunks)

    embedding_provider.embed.assert_not_called()
    vector_store.upsert.assert_not_called()


def test_index_rejects_embedding_count_mismatch(tmp_path):
    document = make_file("# Python", tmp_path / "python.md")
    chunks = split_file(document)
    service, embedding_provider, vector_store = make_service([])

    with pytest.raises(ValueError, match="向量数量与 Chunk 数量不一致"):
        service.index([document], chunks)

    embedding_provider.embed.assert_called_once_with(
        [chunk.content for chunk in chunks]
    )
    vector_store.upsert.assert_not_called()


def test_index_matches_each_chunk_to_its_own_document(tmp_path):
    first_document = make_file("# Python", tmp_path / "python.md")
    second_document = make_file("# Java", tmp_path / "java.md")
    chunks = [*split_file(first_document), *split_file(second_document)]
    vectors = [[1.0, 0.0], [0.0, 1.0]]
    service, _, vector_store = make_service(vectors)

    service.index([second_document, first_document], chunks)

    records = vector_store.upsert.call_args.args[0]
    assert records[0].document_id == first_document.id
    assert records[0].metadata.path == first_document.path
    assert records[1].document_id == second_document.id
    assert records[1].metadata.path == second_document.path
