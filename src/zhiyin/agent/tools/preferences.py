import json
from dataclasses import dataclass

from zhiyin.agent.preferences import (
    ForgetPreferenceArguments,
    RememberPreferenceArguments,
)

from ..context.budget import ContextBudgetExceeded
from ..context.tool_output import ToolOutputBudget
from ..models import ToolExecutionResult
from ..runtime import AgentContext, ToolDependencies


@dataclass
class PreferenceTool:
    forget: bool = False

    @property
    def name(self):
        return "forget_preference" if self.forget else "remember_preference"

    @property
    def arguments_model(self):
        return ForgetPreferenceArguments if self.forget else RememberPreferenceArguments

    @property
    def definition(self):
        return {
            "type": "function",
            "name": self.name,
            "strict": True,
            "description": "删除用户明确要求忘记的偏好。"
            if self.forget
            else (
                "仅保存当前用户明确表达的长期偏好。response_language: zh-CN/en/ja/ko/fr/de/es；"
                "answer_detail: concise/balanced/detailed；code_language: 编程语言名称，例如 Python。"
                "不保存仅针对本轮的要求，不从历史或资料推断偏好。"
            ),
            "parameters": self.arguments_model.model_json_schema(),
        }

    async def invoke(
        self,
        arguments: str,
        context: AgentContext,
        dependencies: ToolDependencies,
        *,
        output_budget: ToolOutputBudget | None = None,
    ):
        args = self.arguments_model.model_validate_json(arguments)
        if dependencies.preference_service is None:
            raise RuntimeError("未配置偏好服务")
        # Check the acknowledgement fits before committing a side effect.
        expected = {"status": "deleted" if self.forget else "saved", "key": args.key}
        if not self.forget:
            expected["value"] = args.value
        if output_budget is not None and not output_budget.fits(
            json.dumps(expected, ensure_ascii=False)
        ):
            raise ContextBudgetExceeded("剩余预算不足以确认偏好修改")
        output = await dependencies.preference_service.change(
            context.owner_id, context.turn_id, context.user_input, args
        )
        return ToolExecutionResult(output)
