from dataclasses import dataclass

from interview_ai.config import config

DEFAULT_SAFETY_MARGIN_TOKENS = 4096
DEFAULT_MAX_OUTPUT_TOKENS = 8192


class ContextBudgetExceeded(ValueError):
    """无法在保留当前问题和必要请求结构的前提下满足预算。"""


class InputTooLarge(ContextBudgetExceeded):
    """当前问题及必要请求结构无法放入输入预算。"""


class CompactionError(RuntimeError):
    """摘要生成失败或未满足压缩要求。"""


@dataclass(frozen=True)
class ContextBudget:
    total_tokens: int
    output_tokens: int | None = None
    safety_margin_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.total_tokens <= 0:
            raise ValueError("总预算必须为正数")
        if self.output_tokens is None:
            object.__setattr__(
                self,
                "output_tokens",
                min(DEFAULT_MAX_OUTPUT_TOKENS, max(1, self.total_tokens // 4)),
            )
        if self.safety_margin_tokens is None:
            object.__setattr__(
                self,
                "safety_margin_tokens",
                min(DEFAULT_SAFETY_MARGIN_TOKENS, self.total_tokens // 8),
            )
        if self.output_tokens <= 0 or self.safety_margin_tokens < 0:
            raise ValueError("输出预算必须为正数，安全余量不能为负数")
        if self.input_tokens <= 0:
            raise ValueError("模型上下文总预算不足以容纳输出预留和安全余量")

    @property
    def input_tokens(self) -> int:
        return self.total_tokens - self.output_tokens - self.safety_margin_tokens

    @classmethod
    def configured(cls) -> "ContextBudget":
        return cls(total_tokens=config.chat_context_budget)
