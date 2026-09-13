import json
from pathlib import Path
from uuid import UUID

import pytest

from interview_ai.evaluation.dataset import load_retrieval_cases
from interview_ai.evaluation.evaluator import evaluate_retrieval
from interview_ai.evaluation.models import RetrievalCase
from interview_ai.indexing.models import SearchResult


class _Searcher:
    def __init__(self, results: dict[str, list[SearchResult]]) -> None:
        self._results = results

    async def search(
        self, query: str, owner_id: str, limit: int = 5
    ) -> list[SearchResult]:
        assert owner_id == "evaluation-user"
        return self._results[query][:limit]


def _result(
    number: int,
    *,
    document: str,
    heading: str,
    content: str,
) -> SearchResult:
    return SearchResult(
        chunk_id=UUID(f"550e8400-e29b-41d4-a716-{number:012d}"),
        document_id=UUID("550e8400-e29b-41d4-a716-446655440099"),
        document_name=document,
        headings=(heading,),
        content=content,
        start_line=number,
        end_line=number + 1,
        score=1.0 - number / 100,
    )


@pytest.mark.asyncio
async def test_evaluate_retrieval_calculates_rank_metrics() -> None:
    cases = [
        RetrievalCase.model_validate(
            {
                "id": "q1",
                "query": "first",
                "relevant_sources": [
                    {"document": "a.md", "heading": "A", "contains": "answer A"}
                ],
            }
        ),
        RetrievalCase.model_validate(
            {
                "id": "q2",
                "query": "second",
                "relevant_sources": [
                    {"document": "b.md", "heading": "B", "contains": "answer B"}
                ],
            }
        ),
    ]
    searcher = _Searcher(
        {
            "first": [
                _result(
                    1,
                    document="a.md",
                    heading="A { #heading-a }",
                    content="answer A",
                )
            ],
            "second": [
                _result(2, document="a.md", heading="A", content="not relevant"),
                _result(3, document="b.md", heading="B", content="answer B"),
            ],
        }
    )

    report = await evaluate_retrieval(
        searcher,
        cases,
        owner_id="evaluation-user",
        embedding_model="test-model",
        dimensions=3,
        limit=5,
    )

    assert report.question_count == 2
    assert report.metrics.hit_at_1 == pytest.approx(0.5)
    assert report.metrics.hit_at_k == pytest.approx(1.0)
    assert report.metrics.mrr_at_k == pytest.approx(0.75)
    assert report.metrics.recall_at_k == pytest.approx(1.0)
    assert [case.first_relevant_rank for case in report.cases] == [1, 2]


def test_load_retrieval_cases_rejects_duplicate_ids(tmp_path: Path) -> None:
    case = {
        "id": "duplicate",
        "query": "question",
        "relevant_sources": [
            {"document": "a.md", "heading": "A", "contains": "answer"}
        ],
    }
    path = tmp_path / "questions.jsonl"
    path.write_text(
        f"{json.dumps(case)}\n{json.dumps(case)}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ID 重复"):
        load_retrieval_cases(path)
