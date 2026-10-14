"""Adapt LangChain node updates to the application's conversation events."""

import asyncio
import logging
from collections.abc import AsyncGenerator, Sequence
from contextlib import aclosing
from dataclasses import replace
from typing import Self
from uuid import uuid4

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI
from openai import DefaultAsyncHttpxClient, DefaultHttpxClient
from pydantic import SecretStr

from interview_ai.agent.context.budget import ContextBudget, ContextBudgetExceeded
from interview_ai.agent.context.builder import INSTRUCTIONS
from interview_ai.agent.context.estimator import estimate_request
from interview_ai.agent.context.tool_output import (
    DEFAULT_MAX_TOOL_RESULT_TOKENS,
    ToolOutputBudget,
)
from interview_ai.agent.models import (
    AgentEvent,
    AgentMessage,
    AssistantMessageEvent,
    AssistantTextDeltaEvent,
    FunctionCallEvent,
    FunctionCallOutputEvent,
    ModelCompletedEvent,
    RetrievedSource,
)
from interview_ai.agent.providers.openai import (
    ModelResponseError,
    OpenAIProvider,
    ToolCallLimitExceeded,
)
from interview_ai.agent.runtime import (
    AgentContext,
    LangChainAgentContext,
    ToolDependencies,
)
from interview_ai.agent.tools.langchain_preferences import (
    forget_preference,
    remember_preference,
)
from interview_ai.agent.tools.search_knowledge import search_knowledgebase
from interview_ai.config import config

logger = logging.getLogger(__name__)


def _input_items(messages: Sequence) -> list[dict]:
    # Artifacts and usage are application data, not part of model input.
    return [
        message.model_dump(
            mode="json",
            exclude={"artifact", "usage_metadata", "response_metadata", "id"},
            exclude_none=True,
        )
        for message in messages
    ]


def _validate_message(message: AIMessage) -> None:
    metadata = message.response_metadata
    if (
        metadata.get("status") in {"failed", "incomplete", "cancelled"}
        or metadata.get("finish_reason") in {"length", "content_filter"}
        or metadata.get("incomplete_details")
        or message.invalid_tool_calls
    ):
        raise ModelResponseError("模型响应未正常完成，请重试")


class _ExecutionMiddleware(AgentMiddleware):
    """Per-generation limits; serialize tools sharing one database session."""

    def __init__(self, budget: ContextBudget, tools: list, max_tool_calls: int):
        self.budget = budget
        self.definitions = tools
        self.max_tool_calls = max_tool_calls
        self.tool_lock = asyncio.Lock()
        self.allocations: dict[str, ToolOutputBudget] = {}

    async def awrap_model_call(self, request, handler):
        instructions = (
            str(request.system_message.text) if request.system_message else None
        )
        items = _input_items(request.messages)
        if (
            estimate_request(instructions, items, self.definitions)
            > self.budget.input_tokens
        ):
            raise ContextBudgetExceeded("模型请求超过输入预算")
        response = await handler(request)
        calls = []
        for message in response.result:
            if isinstance(message, AIMessage):
                _validate_message(message)
                calls.extend(message.tool_calls)
        if len(self.allocations) + len(calls) > self.max_tool_calls:
            raise ToolCallLimitExceeded(f"工具调用次数超过限制：{self.max_tool_calls}")
        if calls:
            ids = [call.get("id") for call in calls]
            if any(
                not call_id or call_id in self.allocations for call_id in ids
            ) or len(set(ids)) != len(ids):
                raise ModelResponseError("工具调用缺少唯一 call_id")
            remaining = self.budget.input_tokens - estimate_request(
                instructions, [*items, *_input_items(response.result)], self.definitions
            )
            # Reserve extra LangChain ToolMessage framing beyond Responses items.
            minima = [
                estimate_request(
                    None,
                    [ToolOutputBudget(0, call["id"]).item('{"error":"tool_failed"}')],
                )
                + 128
                for call in calls
            ]
            extra = remaining - sum(minima)
            if extra < 0:
                raise ContextBudgetExceeded("剩余预算不足以容纳工具结果")
            for call, minimum in zip(calls, minima, strict=True):
                self.allocations[call["id"]] = ToolOutputBudget(
                    min(
                        minimum + extra // len(calls),
                        max(
                            minimum,
                            min(
                                DEFAULT_MAX_TOOL_RESULT_TOKENS,
                                self.budget.input_tokens // 8,
                            ),
                        ),
                    ),
                    call["id"],
                )
        return response

    async def awrap_tool_call(self, request, handler):
        call = request.tool_call
        budget = self.allocations[call["id"]]
        runtime = replace(
            request.runtime,
            context=replace(request.runtime.context, output_budget=budget),
        )
        async with self.tool_lock:
            try:
                result = await handler(replace(request, runtime=runtime))
            except ContextBudgetExceeded:
                raise
            except Exception:
                logger.exception("工具执行失败：%s", call["name"])
                result = ToolMessage(
                    content='{"error":"tool_execution_failed"}',
                    tool_call_id=call["id"],
                    name=call["name"],
                    status="error",
                )
        if not isinstance(result, ToolMessage):
            raise ModelResponseError("工具未返回有效 ToolMessage")
        if result.status == "error":
            result = ToolMessage(
                content='{"error":"tool_execution_failed"}',
                tool_call_id=call["id"],
                name=call["name"],
                status="error",
            )
        if not budget.fits(str(result.text)):
            raise ContextBudgetExceeded("工具结果超过分配的输入预算")
        return result


class LangChainProvider:
    def __init__(
        self,
        model: BaseChatModel | None = None,
        budget: ContextBudget | None = None,
        max_tool_calls: int | None = None,
    ) -> None:
        self._budget = budget or ContextBudget.configured()
        self._owns_model = model is None
        self._model = (
            model
            if model is not None
            else ChatOpenAI(
                model=config.chat_model,
                api_key=SecretStr(config.chat_api_key),
                base_url=config.chat_api_host,
                timeout=60,
                max_tokens=self._budget.output_tokens,
                use_responses_api=True,
                output_version="responses/v1",
                # LangChain caches default HTTP clients. This request-scoped
                # provider must own its clients so closing it cannot break others.
                http_client=DefaultHttpxClient(),
                http_async_client=DefaultAsyncHttpxClient(),
            )
        )
        self._max_tool_calls = (
            config.max_tool_calls if max_tool_calls is None else max_tool_calls
        )
        if self._max_tool_calls < 1:
            raise ValueError("max_tool_calls 必须大于 0")
        self._summary_provider: OpenAIProvider | None = None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._summary_provider is not None:
            await self._summary_provider.aclose()
        if self._owns_model:
            await self._model.root_async_client.close()
            self._model.root_client.close()

    def tool_definitions(self) -> list:
        return [
            {
                "type": "function",
                **convert_to_openai_tool(tool, strict=True)["function"],
            }
            for tool in [search_knowledgebase, remember_preference, forget_preference]
        ]

    async def summarize_history(
        self, messages: Sequence[AgentMessage], *, max_output_tokens: int
    ) -> str:
        # Reuse the existing bounded, tool-free Responses summary implementation.
        if self._summary_provider is None:
            self._summary_provider = OpenAIProvider(budget=self._budget)
        return await self._summary_provider.summarize_history(
            messages, max_output_tokens=max_output_tokens
        )

    async def generate(
        self,
        messages: Sequence[AgentMessage],
        context: AgentContext,
        dependencies: ToolDependencies,
        instructions: str | None = None,
        *,
        stream: bool = False,
    ) -> AsyncGenerator[AgentEvent, None]:
        middleware = _ExecutionMiddleware(
            self._budget, self.tool_definitions(), self._max_tool_calls
        )
        agent = create_agent(
            model=self._model,
            tools=[search_knowledgebase, remember_preference, forget_preference],
            system_prompt=INSTRUCTIONS if instructions is None else instructions,
            context_schema=LangChainAgentContext,
            middleware=[middleware],
        )
        runtime = LangChainAgentContext(
            owner_id=context.owner_id,
            search_service=dependencies.search_service,
            output_budget=None,
            preference_service=dependencies.preference_service,
            turn_id=context.turn_id,
            user_input=context.user_input,
        )
        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        seen = set()
        calls: dict[str, tuple[str, str | None]] = {}
        pending = set()
        last_message = None
        stream_id = str(uuid4())
        events = agent.astream(
            {"messages": [{"role": m.role, "content": m.content} for m in messages]},
            context=runtime,
            config={"recursion_limit": 4 * self._max_tool_calls + 10},
            stream_mode=["messages", "updates"],
        )
        async with aclosing(events):
            async for mode, data in events:
                if mode == "messages":
                    chunk, metadata = data
                    if stream and isinstance(chunk, AIMessageChunk) and chunk.text:
                        key = (
                            f"{stream_id}/{metadata.get('langgraph_checkpoint_ns', '')}"
                            f"/{metadata.get('langgraph_step', '')}/{metadata.get('langgraph_node', '')}"
                        )
                        yield AssistantTextDeltaEvent(
                            stream_key=key,
                            provider_response_id=chunk.response_metadata.get("id"),
                            delta=str(chunk.text),
                        )
                    continue
                if mode != "updates":
                    continue
                for update in data.values():
                    if not isinstance(update, dict):
                        continue
                    for message in update.get("messages", []):
                        if not isinstance(message, (AIMessage, ToolMessage)):
                            continue
                        key = (message.type, message.id or id(message))
                        if key in seen:
                            continue
                        seen.add(key)
                        last_message = message
                        if isinstance(message, AIMessage):
                            _validate_message(message)
                            response_id = message.response_metadata.get("id")
                            for field in usage:
                                usage[field] += (message.usage_metadata or {}).get(
                                    field, 0
                                )
                            if message.text:
                                yield AssistantMessageEvent(
                                    provider_response_id=response_id,
                                    text=str(message.text),
                                )
                            for call in message.tool_calls:
                                call_id = call["id"]
                                calls[call_id] = (call["name"], response_id)
                                pending.add(call_id)
                                yield FunctionCallEvent(
                                    provider_response_id=response_id,
                                    call_id=call_id,
                                    tool_name=call["name"],
                                    arguments=call["args"],
                                )
                        else:
                            call_id = message.tool_call_id
                            if call_id not in pending:
                                raise ModelResponseError("工具结果缺少对应调用")
                            pending.remove(call_id)
                            name, response_id = calls[call_id]
                            artifact = message.artifact or {}
                            sources = (
                                artifact.get("sources", [])
                                if message.status == "success"
                                else []
                            )
                            yield FunctionCallOutputEvent(
                                provider_response_id=response_id,
                                call_id=call_id,
                                tool_name=name,
                                output=str(message.text),
                                succeeded=message.status == "success",
                                sources=tuple(
                                    RetrievedSource.model_validate(s) for s in sources
                                ),
                            )
        if (
            not isinstance(last_message, AIMessage)
            or last_message.tool_calls
            or pending
            or not str(last_message.text).strip()
        ):
            raise ModelResponseError("Agent 未产生有效的最终回答")
        yield ModelCompletedEvent(
            provider="openai",
            model=config.chat_model,
            provider_response_id=last_message.response_metadata.get("id"),
            **usage,
        )
