import json
from pathlib import Path

from pydantic import ValidationError

from .models import RetrievalCase


def load_retrieval_cases(file_path: Path) -> list[RetrievalCase]:
    """从 JSONL 文件加载检索评测问题，并验证 ID 唯一性。"""
    cases: list[RetrievalCase] = []
    ids: set[str] = set()

    for line_number, line in enumerate(
        file_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            case = RetrievalCase.model_validate(json.loads(line))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ValueError(f"评测数据第 {line_number} 行无效：{exc}") from exc
        if case.id in ids:
            raise ValueError(f"评测问题 ID 重复：{case.id}")
        ids.add(case.id)
        cases.append(case)

    if not cases:
        raise ValueError("评测问题集不能为空")
    return cases
