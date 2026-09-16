from collections.abc import AsyncIterator, Sequence
from typing import Protocol

from openai.types.responses.function_tool_param import FunctionToolParam

from .models import AgentEvent, AgentMessage, ToolExecutionResult
from .runtime import AgentContext, ToolDependencies


# 上下文服务，包含上下文管理，上下文压缩等功能
class ContextService(Protocol):
    async def compact(self): ...


# 工具接口，后续接入mcp
class ToolProvider(Protocol): ...


class AgentTool(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def definition(self) -> FunctionToolParam:
        """提供给 Responses API 的工具定义。"""
        ...

    async def invoke(
        self, arguments: str, context: AgentContext, dependencies: ToolDependencies
    ) -> ToolExecutionResult:
        """执行模型发起的工具调用，返回 JSON 字符串。"""
        ...


class AgentProvider(Protocol):
    def generate(
        self,
        messages: Sequence[AgentMessage],
        context: AgentContext,
        dependencies: ToolDependencies,
    ) -> AsyncIterator[AgentEvent]: ...


# class ChatInput(Protocol):


# class AIService(Protocol):
#     async def chat(self,message:Message): Stream[ResponseStreamEvent]
