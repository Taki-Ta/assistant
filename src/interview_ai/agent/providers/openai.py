import json
import logging
from collections.abc import AsyncIterator, Sequence
from types import TracebackType
from typing import Self

from openai import AsyncOpenAI
from openai.types.responses import EasyInputMessageParam, ResponseInputParam
from openai.types.responses.function_tool_param import FunctionToolParam

from interview_ai.agent.context.budget import ContextBudget, ContextBudgetExceeded
from interview_ai.agent.context.compactor import COMPACTION_INSTRUCTIONS
from interview_ai.agent.context.estimator import estimate_request
from interview_ai.agent.tools.registry import ToolRegistry
from interview_ai.config import config

from ..models import (
    AgentEvent,
    AgentMessage,
    AssistantMessageEvent,
    FunctionCallEvent,
    FunctionCallOutputEvent,
    ModelCompletedEvent,
    ToolExecutionResult,
)
from ..runtime import AgentContext, ToolDependencies
from ..tools.default_registry import tool_registry

logger = logging.getLogger(__name__)


class ToolCallLimitExceeded(RuntimeError):
    pass


class ModelResponseError(RuntimeError):
    """模型未正常完成生成，不能作为完整回答保存。"""


def _validate_completion(response: object) -> None:
    # 部分兼容服务不返回 status；有状态时必须明确完成。
    response_status = getattr(response, "status", None)
    if response_status is not None and response_status != "completed":
        raise ModelResponseError("模型响应未正常完成，请重试")


class OpenAIProvider:
    def __init__(
        self,
        client: AsyncOpenAI | None = None,
        tools: ToolRegistry | None = None,
        max_tool_calls: int | None = None,
        budget: ContextBudget | None = None,
    ) -> None:
        self._owns_client = client is None
        self._client = client or AsyncOpenAI(
            api_key=config.chat_api_key,
            base_url=config.chat_api_host,
        )

        self._tools = tools if tools is not None else tool_registry
        self._budget = budget or ContextBudget.configured()

        self._max_tool_calls = (
            max_tool_calls if max_tool_calls is not None else config.max_tool_calls
        )
        if self._max_tool_calls < 1:
            raise ValueError("max_tool_calls 必须大于 0")

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.close()

    def tool_definitions(self) -> list[FunctionToolParam]:
        return self._tools.definitions()

    async def summarize_history(
        self, messages: Sequence[AgentMessage], *, max_output_tokens: int
    ) -> str:
        instructions = (
            f"{COMPACTION_INSTRUCTIONS}\n摘要最多约 {max_output_tokens} Token。"
        )
        input_items = [
            {"role": item.role, "content": item.content} for item in messages
        ]
        if estimate_request(instructions, input_items) > (
            self._budget.total_tokens
            - self._budget.safety_margin_tokens
            - max_output_tokens
        ):
            raise ContextBudgetExceeded("历史摘要请求超过预算")
        response = await self._client.responses.create(
            model=config.chat_model,
            instructions=instructions,
            input=input_items,
            max_output_tokens=max_output_tokens,
        )
        _validate_completion(response)
        if not response.output_text.strip():
            raise ModelResponseError("模型未返回有效摘要")
        return response.output_text

    async def generate(
        self,
        messages: Sequence[AgentMessage],
        context: AgentContext,
        dependencies: ToolDependencies,
        instructions: str | None = None,
    ) -> AsyncIterator[AgentEvent]:
        input_items: ResponseInputParam = [
            EasyInputMessageParam(role=message.role, content=message.content)
            for message in messages
        ]
        tool_call_count = 0
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        tools = self.tool_definitions()

        while True:
            estimated_tokens = estimate_request(instructions, input_items, tools)
            logger.debug(
                "模型请求输入 Token 估算：%d",
                estimated_tokens,
            )
            if estimated_tokens > self._budget.input_tokens:
                raise ContextBudgetExceeded(
                    "模型请求超过输入预算，可能包含过长的工具结果"
                )
            response = await self._client.responses.create(
                model=config.chat_model,
                input=list(input_items),
                tools=tools,
                instructions=instructions,
                max_output_tokens=self._budget.output_tokens,
            )

            usage = getattr(response, "usage", None)
            if usage is not None:
                input_tokens += usage.input_tokens
                output_tokens += usage.output_tokens
                total_tokens += usage.total_tokens

            _validate_completion(response)
            input_items.extend(response.output)  # type: ignore[arg-type]

            has_tool_call = False

            for item in response.output:
                if item.type != "function_call":
                    continue

                has_tool_call = True
                tool_call_count += 1
                if tool_call_count > self._max_tool_calls:
                    raise ToolCallLimitExceeded(
                        f"工具调用次数超过限制：{self._max_tool_calls}"
                    )

                yield FunctionCallEvent(
                    provider_response_id=getattr(response, "id", None),
                    provider_item_id=getattr(item, "id", None),
                    call_id=item.call_id,
                    arguments=json.loads(item.arguments),
                    tool_name=item.name,
                )

                try:
                    result = await self._tools.invoke(
                        item.name,
                        item.arguments,
                        context,
                        dependencies,
                    )
                    tool_succeeded = True
                except Exception:
                    logger.exception("工具执行失败：%s", item.name)
                    result = ToolExecutionResult(
                        json.dumps(
                            {
                                "error": {
                                    "code": "tool_execution_failed",
                                    "message": f"工具 {item.name} 执行失败",
                                }
                            },
                            ensure_ascii=False,
                        ),
                        (),
                    )
                    tool_succeeded = False

                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": result.output,
                    }
                )
                yield FunctionCallOutputEvent(
                    provider_response_id=getattr(response, "id", None),
                    call_id=item.call_id,
                    tool_name=item.name,
                    succeeded=tool_succeeded,
                    output=result.output,
                    sources=result.sources,
                )

            if not has_tool_call:
                if not response.output_text.strip():
                    raise ModelResponseError("模型未返回有效回答")
                message_item = next(
                    (item for item in response.output if item.type == "message"),
                    None,
                )
                yield AssistantMessageEvent(
                    provider_response_id=getattr(response, "id", None),
                    provider_item_id=getattr(message_item, "id", None),
                    text=response.output_text,
                )
                yield ModelCompletedEvent(
                    provider_response_id=getattr(response, "id", None),
                    provider="openai",
                    model=config.chat_model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                )
                return
