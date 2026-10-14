from dataclasses import dataclass

from .estimator import estimate_request

DEFAULT_MAX_TOOL_RESULT_TOKENS = 2048


@dataclass(frozen=True)
class ToolOutputBudget:
    """单个工具结果的容量，包含输出 Item 的结构及 JSON 转义开销。"""

    max_tokens: int
    call_id: str

    def item(self, output: str) -> dict[str, str]:
        return {
            "type": "function_call_output",
            "call_id": self.call_id,
            "output": output,
        }

    def fits(self, output: str) -> bool:
        return estimate_request(None, [self.item(output)]) <= self.max_tokens
