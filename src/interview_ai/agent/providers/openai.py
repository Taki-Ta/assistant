import json
import logging
from types import TracebackType
from typing import Self

from openai import AsyncOpenAI
from openai.types.responses import Response, ResponseInputParam

from interview_ai.agent.tools.registry import ToolRegistry
from interview_ai.config import config

from ..runtime import AgentContext, ToolDependencies
from ..tools.default_registry import tool_registry
from ..models import AgentResult, ToolExecutionResult

logger = logging.getLogger(__name__)


class ToolCallLimitExceeded(RuntimeError):
    pass


class OpenAIProvider:
    def __init__(
        self,
        client: AsyncOpenAI | None = None,
        tools: ToolRegistry | None = None,
        messages: ResponseInputParam | None = None,
        max_tool_calls: int | None = None,
    ) -> None:
        self._owns_client = client is None
        self._client = client or AsyncOpenAI(
            api_key=config.chat_api_key,
            base_url=config.chat_api_host,
        )

        self._tools = tools if tools is not None else tool_registry

        self._messages = messages if messages is not None else []
        self._max_tool_calls = (
            max_tool_calls if max_tool_calls is not None else config.max_tool_calls
        )
        self._sources = ()

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

    async def generate(
        self,
        prompt: str | None,
        context: AgentContext,
        dependencies: ToolDependencies,
    ) -> AgentResult:

        self._sources = ()

        # 附带工具的请求
        if prompt:
            self._messages.append({"role": "user", "content": prompt})
        response = await self.handle_response(context, dependencies)
        sources_by_chunk_id = {source.chunk_id: source for source in self._sources}
        return AgentResult(response.output_text, tuple(sources_by_chunk_id.values()))

    async def handle_response(
        self,
        context: AgentContext,
        dependencies: ToolDependencies,
    ) -> Response:
        tool_call_count = 0

        while True:
            response = await self._client.responses.create(
                model=config.chat_model,
                input=self._messages,  # type: ignore
                tools=self._tools.definitions(),
            )

            self._messages.extend(response.output)  # type: ignore

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

                try:
                    result = await self._tools.invoke(
                        item.name,
                        item.arguments,
                        context,
                        dependencies,
                    )
                    self._sources += result.sources
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

                self._messages.append(
                    {
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": result.output,
                    }
                )

            if not has_tool_call:
                return response
