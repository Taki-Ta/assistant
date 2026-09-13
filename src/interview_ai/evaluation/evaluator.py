import math
import re
from datetime import UTC, datetime
from time import perf_counter
from typing import Protocol

from interview_ai.indexing.models import SearchResult

from .models import (
    EvaluatedMatch,
    RetrievalCase,
    RetrievalCaseResult,
    RetrievalMetrics,
    RetrievalReport,
)


class RetrievalSearcher(Protocol):
    async def search(
        self, query: str, owner_id: str, limit: int = 5
    ) -> list[SearchResult]: ...


HEADING_ATTRIBUTE_PATTERN = re.compile(r"\s+\{\s*#[^}]+\}\s*$")


def _normalize_heading(heading: str) -> str:
    return HEADING_ATTRIBUTE_PATTERN.sub("", heading).strip()


def _matching_source_indexes(case: RetrievalCase, result: SearchResult) -> set[int]:
    normalized_headings = {_normalize_heading(item) for item in result.headings}
    return {
        index
        for index, source in enumerate(case.relevant_sources)
        if result.document_name == source.document
        and source.heading in normalized_headings
        and source.contains in result.content
    }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


async def evaluate_retrieval(
    searcher: RetrievalSearcher,
    cases: list[RetrievalCase],
    *,
    owner_id: str,
    embedding_model: str,
    dimensions: int,
    limit: int = 5,
) -> RetrievalReport:
    """运行检索评测并返回包含逐题 Top-K 的机器可读报告。"""
    if not cases:
        raise ValueError("评测问题集不能为空")
    if limit <= 0:
        raise ValueError("limit 必须大于 0")

    case_results: list[RetrievalCaseResult] = []
    reciprocal_ranks: list[float] = []
    recalls: list[float] = []
    latencies: list[float] = []

    for case in cases:
        started_at = perf_counter()
        search_results = await searcher.search(case.query, owner_id, limit)
        latency_ms = (perf_counter() - started_at) * 1000
        latencies.append(latency_ms)

        matched_sources: set[int] = set()
        first_relevant_rank: int | None = None
        evaluated_matches: list[EvaluatedMatch] = []
        for rank, result in enumerate(search_results, start=1):
            source_indexes = _matching_source_indexes(case, result)
            relevant = bool(source_indexes)
            matched_sources.update(source_indexes)
            if relevant and first_relevant_rank is None:
                first_relevant_rank = rank
            evaluated_matches.append(
                EvaluatedMatch(
                    rank=rank,
                    chunk_id=result.chunk_id,
                    document_id=result.document_id,
                    document_name=result.document_name,
                    headings=result.headings,
                    content=result.content,
                    start_line=result.start_line,
                    end_line=result.end_line,
                    score=result.score,
                    relevant=relevant,
                )
            )

        recall = len(matched_sources) / len(case.relevant_sources)
        recalls.append(recall)
        reciprocal_ranks.append(
            1.0 / first_relevant_rank if first_relevant_rank is not None else 0.0
        )
        case_results.append(
            RetrievalCaseResult(
                id=case.id,
                query=case.query,
                tags=case.tags,
                latency_ms=latency_ms,
                first_relevant_rank=first_relevant_rank,
                recall_at_k=recall,
                results=evaluated_matches,
            )
        )

    count = len(case_results)
    return RetrievalReport(
        created_at=datetime.now(UTC),
        embedding_model=embedding_model,
        dimensions=dimensions,
        limit=limit,
        question_count=count,
        metrics=RetrievalMetrics(
            hit_at_1=sum(item.first_relevant_rank == 1 for item in case_results)
            / count,
            hit_at_k=sum(item.first_relevant_rank is not None for item in case_results)
            / count,
            mrr_at_k=sum(reciprocal_ranks) / count,
            recall_at_k=sum(recalls) / count,
            average_latency_ms=sum(latencies) / count,
            p95_latency_ms=_percentile(latencies, 0.95),
        ),
        cases=case_results,
    )
