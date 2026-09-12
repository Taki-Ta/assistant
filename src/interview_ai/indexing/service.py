from dataclasses import dataclass

from interview_ai.core.protocols import IndexableDocument
from interview_ai.db.models import Chunk, VectorRecord

from .protocols import ChunkStore, EmbeddingProvider, VectorStore


@dataclass
class IndexService:
    embedding_provider: EmbeddingProvider
    chunk_store: ChunkStore
    vector_store: VectorStore

    async def index(
        self,
        document: IndexableDocument,
        chunks: list[Chunk],
    ) -> int:
        """为一批 Chunk 生成向量并写入向量存储。"""
        if any(chunk.document_id != document.id for chunk in chunks):
            raise ValueError("Chunk 找不到对应 Document")

        contents = [chunk.content for chunk in chunks]
        vectors = await self.embedding_provider.embed(contents) if contents else []
        if len(vectors) != len(contents):
            raise ValueError(
                "向量数量与 Chunk 数量不一致："
                f"chunks={len(chunks)}, vectors={len(vectors)}"
            )

        records = [
            VectorRecord(
                chunk_id=chunk.id,
                vector=vector,
                embedding_model=self.embedding_provider.model_name,
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]

        await self.chunk_store.replace(document.id, chunks)
        await self.vector_store.upsert(records)
        return len(records)
