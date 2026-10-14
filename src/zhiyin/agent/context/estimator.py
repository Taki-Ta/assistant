"""无模型依赖的文本请求估算；结果不是供应商的精确 Token 计数。"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from math import ceil

from pydantic import BaseModel

STRUCTURE_OVERHEAD = 4


def estimate_text(text: str) -> int:
    ascii_count = sum(character.isascii() for character in text)
    return ceil(ascii_count / 3) + len(text) - ascii_count


def _plain(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_none=True)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "__dict__"):
        return _plain(vars(value))
    return value


def estimate_request(
    instructions: str | None,
    input_items: Sequence[object],
    tools: Sequence[object] = (),
) -> int:
    """覆盖指令、消息、输出 Item 和工具定义；JSON 仅用于结构开销估算。"""
    tokens = estimate_text(instructions) + STRUCTURE_OVERHEAD if instructions else 0
    for item in input_items:
        serialized = json.dumps(_plain(item), ensure_ascii=False, separators=(",", ":"))
        tokens += estimate_text(serialized) + STRUCTURE_OVERHEAD
    if tools:
        serialized = json.dumps(
            _plain(tools), ensure_ascii=False, separators=(",", ":")
        )
        tokens += estimate_text(serialized) + STRUCTURE_OVERHEAD
    return tokens
