import hashlib
from datetime import UTC, datetime
from pathlib import Path

from openai import AsyncOpenAI
from sqlalchemy import delete

from interview_ai.config import config
from interview_ai.db.database import session_factory
from interview_ai.db.models import Document, DocumentStatus
from interview_ai.db.repositories import PostgresChunkStore
from interview_ai.indexing.embedding import OpenAIEmbeddingProvider
from interview_ai.indexing.search_service import SearchService
from interview_ai.indexing.service import IndexService
from interview_ai.indexing.vector_store import PostgresVectorStore
from interview_ai.ingestion.scanner import split_document

from .dataset import load_retrieval_cases
from .evaluator import evaluate_retrieval
from .models import RetrievalReport

EVALUATION_OWNER_ID = "__retrieval_evaluation__"


async def run_retrieval_evaluation(
    dataset_root: Path,
    *,
    limit: int = 5,
) -> RetrievalReport:
    """重建隔离的评测语料索引，并使用真实 Embedding 和 pgvector 评测。"""
    questions_path = dataset_root / "questions.jsonl"
    corpus_root = dataset_root / "corpus"
    cases = load_retrieval_cases(questions_path)
    corpus_paths = sorted(corpus_root.glob("*.md"))
    if not corpus_paths:
        raise ValueError(f"评测语料目录中没有 Markdown 文件：{corpus_root}")

    async with AsyncOpenAI(api_key=config.api_key, base_url=config.host) as client:
        embedding_provider = OpenAIEmbeddingProvider(
            client=client,
            model=config.embedding_model_name,
            dimensions=config.dimensions,
            batch_size=config.embedding_batch_size,
        )
        async with session_factory() as db:
            vector_store = PostgresVectorStore(db)
            index_service = IndexService(
                embedding_provider=embedding_provider,
                chunk_store=PostgresChunkStore(db),
                vector_store=vector_store,
            )
            search_service = SearchService(
                embedding_provider=embedding_provider,
                vector_store=vector_store,
                default_limit=limit,
                score_threshold=None,
            )

            await db.execute(
                delete(Document).where(Document.owner_id == EVALUATION_OWNER_ID)
            )
            for corpus_path in corpus_paths:
                content_bytes = corpus_path.read_bytes()
                content = content_bytes.decode("utf-8-sig")
                document = Document(
                    owner_id=EVALUATION_OWNER_ID,
                    name=corpus_path.name,
                    content=content,
                    content_hash=hashlib.sha256(content_bytes).hexdigest(),
                    size_bytes=len(content_bytes),
                )
                db.add(document)
                await db.flush()
                document.status = DocumentStatus.INDEXING
                chunks = split_document(document)
                document.chunk_count = await index_service.index(document, chunks)
                document.status = DocumentStatus.INDEXED
                document.indexed_at = datetime.now(UTC)
            await db.commit()

            return await evaluate_retrieval(
                search_service,
                cases,
                owner_id=EVALUATION_OWNER_ID,
                embedding_model=embedding_provider.model_name,
                dimensions=config.dimensions,
                limit=limit,
            )


def save_retrieval_report(report: RetrievalReport, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
