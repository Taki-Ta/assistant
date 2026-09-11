from uuid import UUID

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from interview_ai.db.models import Chunk


class PostgresChunkStore:
    """在同一事务中替换一个文档的全部 Chunk。"""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def replace(self, document_id: UUID, chunks: list[Chunk]) -> None:
        if any(chunk.document_id != document_id for chunk in chunks):
            raise ValueError("Chunk 的 document_id 与待替换文档不一致")

        await self._db.execute(delete(Chunk).where(Chunk.document_id == document_id))
        self._db.add_all(chunks)
        await self._db.flush()
