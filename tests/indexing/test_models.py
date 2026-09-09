from uuid import UUID

import pytest

from interview_ai.indexing.models import InMemoryVectorStore, VectorRecord

TEST_DATA = [
    (
        "550e8400-e29b-41d4-a716-446655440001",
        [0.90, 0.80, 0.10, 0.05, 0.02],
        "Python 是一种流行的编程语言",
    ),
    (
        "550e8400-e29b-41d4-a716-446655440002",
        [0.85, 0.75, 0.15, 0.05, 0.03],
        "Python 可以用于 Web 后端开发",
    ),
    (
        "550e8400-e29b-41d4-a716-446655440003",
        [0.80, 0.70, 0.20, 0.10, 0.05],
        "FastAPI 是一个 Python Web 框架",
    ),
    (
        "550e8400-e29b-41d4-a716-446655440004",
        [0.10, 0.15, 0.90, 0.80, 0.05],
        "苹果是一种常见的水果",
    ),
    (
        "550e8400-e29b-41d4-a716-446655440005",
        [0.05, 0.10, 0.85, 0.90, 0.10],
        "香蕉含有丰富的钾元素",
    ),
    (
        "550e8400-e29b-41d4-a716-446655440006",
        [0.15, 0.10, 0.80, 0.85, 0.05],
        "橙子含有丰富的维生素 C",
    ),
    (
        "550e8400-e29b-41d4-a716-446655440007",
        [0.05, 0.10, 0.10, 0.20, 0.95],
        "猫是一种常见的家庭宠物",
    ),
    (
        "550e8400-e29b-41d4-a716-446655440008",
        [0.10, 0.05, 0.15, 0.20, 0.90],
        "狗是人类常见的伴侣动物",
    ),
    (
        "550e8400-e29b-41d4-a716-446655440009",
        [0.70, 0.65, 0.10, 0.10, 0.20],
        "C# 常用于开发 .NET 应用程序",
    ),
    (
        "550e8400-e29b-41d4-a716-446655440010",
        [0.65, 0.70, 0.10, 0.05, 0.20],
        "Java 是一种面向对象的编程语言",
    ),
]


def make_records() -> list[VectorRecord]:
    return [
        VectorRecord(UUID(chunk_id), vector.copy(), content)
        for chunk_id, vector, content in TEST_DATA
    ]


@pytest.fixture
def records():
    return make_records()


@pytest.fixture
def store(records):
    vector_store = InMemoryVectorStore()
    vector_store.upsert(records)
    return vector_store


def test_upsert_inserts_records_without_duplicates(store, records):
    assert len(store.inner) == 10

    store.upsert(records)

    assert len(store.inner) == 10


def test_upsert_replaces_record_with_same_chunk_id(store, records):
    replacement = VectorRecord(
        chunk_id=records[0].chunk_id,
        vector=[0.0, 0.0, 0.0, 0.0, 1.0],
        content="更新后的内容",
    )

    store.upsert([replacement])

    assert len(store.inner) == 10
    assert store.inner[0] is replacement
    assert store.inner[0].content == "更新后的内容"


def test_delete_removes_existing_records(store, records):
    ids = [item.chunk_id for item in records[:3]]

    store.delete(ids)

    assert len(store.inner) == 7
    assert not set(ids) & {item.chunk_id for item in store.inner}


def test_delete_ignores_unknown_id(store):
    store.delete([UUID("550e8400-e29b-41d4-a716-446655449999")])

    assert len(store.inner) == 10


def test_search_returns_results_ordered_by_similarity(store, records):
    result = store.search(records[0].vector, limit=3)

    assert len(result) == 3
    assert result[0].chunk_id == records[0].chunk_id
    assert result[0].score == pytest.approx(1.0)
    assert [item.score for item in result] == sorted(
        (item.score for item in result), reverse=True
    )


def test_search_limits_result_count(store, records):
    result = store.search(records[0].vector, limit=100)

    assert len(result) == len(records)


def test_search_rejects_different_vector_dimensions(store):
    with pytest.raises(ValueError, match="向量长度不一致"):
        store.search([0.0, 0.1], limit=3)


@pytest.mark.parametrize("limit", [0, -1])
def test_search_rejects_non_positive_limit(store, limit):
    with pytest.raises(ValueError, match="limit 必须大于 0"):
        store.search([1.0, 0.0, 0.0, 0.0, 0.0], limit)


def test_search_empty_store_returns_empty_result(records):
    store = InMemoryVectorStore()

    assert store.search(records[0].vector, limit=3) == []
