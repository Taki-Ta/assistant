from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from interview_ai.db.models import VectorRecord

from ..models import SearchResult


class PGVectorStore:
    """使用 PostgreSQL pgvector 保存和检索 Chunk 向量。"""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def upsert(self, records: list[VectorRecord]) -> None:
        if not records:
            return

        values = [
            {
                "chunk_id": record.chunk_id,
                "vector": record.vector,
                "embedding_model": record.embedding_model,
            }
            for record in records
        ]
        statement = insert(VectorRecord).values(values)
        statement = statement.on_conflict_do_update(
            index_elements=[VectorRecord.chunk_id],
            set_={
                "vector": statement.excluded.vector,
                "embedding_model": statement.excluded.embedding_model,
                "updated_at": func.now(),
            },
        )
        await self._db.execute(statement)

    async def delete(self, chunk_ids: list[UUID]) -> None:
        if not chunk_ids:
            return
        await self._db.execute(
            delete(VectorRecord).where(VectorRecord.chunk_id.in_(chunk_ids))
        )

    async def search(
        self,
        query_vector: list[float],
        limit: int = 5,
    ) -> list[SearchResult]:
        if limit <= 0:
            raise ValueError("limit 必须大于 0")

        distance = VectorRecord.vector.cosine_distance(query_vector)
        statement = (
            select(VectorRecord.chunk_id, distance.label("distance"))
            .order_by(distance)
            .limit(limit)
        )
        rows = (await self._db.execute(statement)).all()
        return [
            SearchResult(chunk_id=chunk_id, score=1.0 - float(distance_value))
            for chunk_id, distance_value in rows
        ]
