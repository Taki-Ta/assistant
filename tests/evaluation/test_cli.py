from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock

from typer.testing import CliRunner

from interview_ai.cli import app
from interview_ai.evaluation.models import (
    RetrievalMetrics,
    RetrievalReport,
)

runner = CliRunner()


def test_eval_retrieval_prints_metrics_and_saves_report(monkeypatch, tmp_path) -> None:
    report = RetrievalReport(
        created_at=datetime.now(UTC),
        embedding_model="test-model",
        dimensions=3,
        limit=5,
        question_count=2,
        metrics=RetrievalMetrics(
            hit_at_1=0.5,
            hit_at_k=1.0,
            mrr_at_k=0.75,
            recall_at_k=1.0,
            average_latency_ms=10.0,
            p95_latency_ms=15.0,
        ),
        cases=[],
    )
    run_evaluation = AsyncMock(return_value=report)
    save_report = Mock()
    monkeypatch.setattr("interview_ai.cli.run_retrieval_evaluation", run_evaluation)
    monkeypatch.setattr("interview_ai.cli.save_retrieval_report", save_report)
    report_path = tmp_path / "report.json"

    result = runner.invoke(
        app,
        [
            "eval-retrieval",
            "--dataset",
            str(tmp_path),
            "--report",
            str(report_path),
        ],
    )

    assert result.exit_code == 0
    assert "Hit@1：50.00%" in result.stdout
    assert "Hit@5：100.00%" in result.stdout
    assert "MRR@5：0.7500" in result.stdout
    run_evaluation.assert_awaited_once_with(tmp_path.resolve(), limit=5)
    save_report.assert_called_once_with(report, report_path)
