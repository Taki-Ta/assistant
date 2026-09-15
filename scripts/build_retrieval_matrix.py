from __future__ import annotations

import argparse
import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HEADING_ATTRIBUTE_PATTERN = re.compile(r"\s+\{\s*#[^}]+\}\s*$")


def normalize_heading(heading: str) -> str:
    return HEADING_ATTRIBUTE_PATTERN.sub("", heading).strip()


def matching_source_indexes(
    sources: list[dict[str, str]], result: dict[str, Any]
) -> set[int]:
    headings = {normalize_heading(item) for item in result["headings"]}
    return {
        index
        for index, source in enumerate(sources)
        if result["document_name"] == source["document"]
        and source["heading"] in headings
        and source["contains"] in result["content"]
    }


def load_questions(path: Path) -> dict[str, dict[str, Any]]:
    questions = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {question["id"]: question for question in questions}


def derive_report(
    raw: dict[str, Any],
    questions: dict[str, dict[str, Any]],
    *,
    limit: int,
    threshold: float,
) -> dict[str, Any]:
    cases = []
    reciprocal_ranks = []
    recalls = []
    latencies = []

    for original in raw["cases"]:
        sources = questions[original["id"]]["relevant_sources"]
        results = []
        matched_sources: set[int] = set()
        first_relevant_rank = None
        for result in original["results"]:
            if result["rank"] > limit or result["score"] < threshold:
                continue
            source_indexes = matching_source_indexes(sources, result)
            result = {**result, "relevant": bool(source_indexes)}
            results.append(result)
            matched_sources.update(source_indexes)
            if source_indexes and first_relevant_rank is None:
                first_relevant_rank = result["rank"]

        recall = len(matched_sources) / len(sources)
        case = {
            **original,
            "first_relevant_rank": first_relevant_rank,
            "recall_at_k": recall,
            "results": results,
        }
        cases.append(case)
        reciprocal_ranks.append(
            1.0 / first_relevant_rank if first_relevant_rank else 0.0
        )
        recalls.append(recall)
        latencies.append(original["latency_ms"])

    count = len(cases)
    ordered_latencies = sorted(latencies)
    p95_index = max(0, math.ceil(0.95 * count) - 1)
    metrics = {
        "hit_at_1": sum(case["first_relevant_rank"] == 1 for case in cases) / count,
        "hit_at_k": sum(case["first_relevant_rank"] is not None for case in cases)
        / count,
        "mrr_at_k": sum(reciprocal_ranks) / count,
        "recall_at_k": sum(recalls) / count,
        "average_latency_ms": sum(latencies) / count,
        "p95_latency_ms": ordered_latencies[p95_index],
    }
    return {
        **raw,
        "created_at": datetime.now(UTC).isoformat(),
        "limit": limit,
        "score_threshold": threshold,
        "metrics": metrics,
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="从最大 Top-K 原始报告派生评测矩阵")
    parser.add_argument("--raw-report", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, nargs="+", default=[3, 5, 7])
    parser.add_argument("--threshold", type=float, nargs="+", default=[0.5, 0.6, 0.7])
    args = parser.parse_args()

    raw = json.loads(args.raw_report.read_text(encoding="utf-8"))
    questions = load_questions(args.questions)
    if max(args.top_k) > raw["limit"]:
        raise SystemExit("派生的 Top-K 不能大于原始报告的 Top-K")

    args.output.mkdir(parents=True, exist_ok=True)
    summary = []
    for limit in args.top_k:
        for threshold in args.threshold:
            report = derive_report(raw, questions, limit=limit, threshold=threshold)
            filename = f"top{limit}-threshold-{threshold:.1f}.json"
            (args.output / filename).write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            summary.append(
                {
                    "top_k": limit,
                    "threshold": threshold,
                    **report["metrics"],
                    "empty_result_count": sum(
                        not case["results"] for case in report["cases"]
                    ),
                    "average_result_count": sum(
                        len(case["results"]) for case in report["cases"]
                    )
                    / report["question_count"],
                    "report": filename,
                }
            )

    summary_path = args.output / "matrix-summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "source_report": args.raw_report.name,
                "method": "截断原始 Top-K 后按余弦相似度阈值过滤；共享原始查询延迟",
                "results": summary,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"已生成 {len(summary)} 组报告：{summary_path}")


if __name__ == "__main__":
    main()
