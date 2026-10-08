from collections.abc import AsyncGenerator, Sequence
from typing import Protocol

from openai.types.responses.function_tool_param import FunctionToolParam

from .context.tool_output import ToolOutputBudget
from .models import AgentEvent, AgentMessage, ToolExecutionResult
from .runtime import AgentContext, ToolDependencies


class AgentTool(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def definition(self) -> FunctionToolParam:
        """提供给 Responses API 的工具定义。"""
        ...

    async def invoke(
        self,
        arguments: str,
        context: AgentContext,
        dependencies: ToolDependencies,
        *,
        output_budget: ToolOutputBudget | None = None,
    ) -> ToolExecutionResult:
        """执行模型发起的工具调用，返回 JSON 字符串。"""
        ...


class AgentProvider(Protocol):
    def tool_definitions(self) -> list[FunctionToolParam]: ...

    def generate(
        self,
        messages: Sequence[AgentMessage],
        context: AgentContext,
        dependencies: ToolDependencies,
        instructions: str | None = None,
        *,
        stream: bool = False,
    ) -> AsyncGenerator[AgentEvent, None]: ...


# class ChatInput(Protocol):


# class AIService(Protocol):
#     async def chat(self,message:Message): Stream[ResponseStreamEvent]
