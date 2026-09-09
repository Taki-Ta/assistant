from dataclasses import dataclass

from interview_ai.ingestion.models import Chunk, Document

from .embedding import EmbeddingProvider
from .models import ChunkMetadata, VectorRecord
from .vector_store import VectorStore


@dataclass
class IndexService:
    embedding_provider: EmbeddingProvider
    vector_store: VectorStore

    def index(self, documents: list[Document], chunks: list[Chunk]) -> int:
        """为一批 Chunk 生成向量并写入向量存储。"""
        if not chunks:
            return 0

        documents_by_id = {document.id: document for document in documents}
        for chunk in chunks:
            if chunk.document_id not in documents_by_id:
                raise ValueError(f"Chunk 找不到对应 Document：{chunk.id}")

        contents = [chunk.content for chunk in chunks]
        vectors = self.embedding_provider.embed(contents)
        if len(vectors) != len(contents):
            raise ValueError(
                "向量数量与 Chunk 数量不一致："
                f"chunks={len(chunks)}, vectors={len(vectors)}"
            )

        records: list[VectorRecord] = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            document = documents_by_id[chunk.document_id]
            records.append(
                VectorRecord(
                    chunk_id=chunk.id,
                    vector=vector,
                    document_id=chunk.document_id,
                    content=chunk.content,
                    metadata=ChunkMetadata(
                        path=document.path,
                        headings=chunk.headings,
                        start_line=chunk.start_line,
                        end_line=chunk.end_line,
                    ),
                )
            )

        self.vector_store.upsert(records)
        return len(records)
