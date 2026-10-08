import logging
from collections.abc import AsyncGenerator

from anyio import CancelScope
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from openai import OpenAIError
from starlette.types import Receive, Scope, Send

from interview_ai.agent.context.budget import (
    CompactionError,
    ContextBudgetExceeded,
    InputTooLarge,
)
from interview_ai.agent.models import ChatStartedEvent, ChatStreamEvent
from interview_ai.agent.providers.openai import (
    ModelResponseError,
    ToolCallLimitExceeded,
)
from interview_ai.agent.services import SessionNotFoundError

from ..dependencies import (
    AgentContextDependency,
    ConversationServiceDependency,
    ToolDependenciesDependency,
)
from ..schemas import ChatErrorEvent, ChatRequest, ChatResponse, SourceResponse

router = APIRouter(prefix="/api/v1", tags=["ai"])


@router.post("/chat")
async def chat(
    input: ChatRequest,
    service: ConversationServiceDependency,
    context: AgentContextDependency,
    dependencies: ToolDependenciesDependency,
) -> ChatResponse:
    try:
        answer = await service.chat(
            input.session_id,
            input.message,
            context,
            dependencies,
        )
    except SessionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        ) from exc
    except InputTooLarge as exc:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail="问题过长，无法放入当前上下文预算，请缩短输入",
        ) from exc
    except ContextBudgetExceeded as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="上下文超过当前预算，无法完成本轮请求",
        ) from exc
    except (
        CompactionError,
        ModelResponseError,
        OpenAIError,
        ToolCallLimitExceeded,
    ) as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="模型处理未正常完成，请稍后重试",
        ) from exc

    return ChatResponse(
        answer=answer.answer,
        retrieved_sources=tuple(
            SourceResponse.model_validate(item) for item in answer.retrieved_sources
        ),
        session_id=answer.session_id,
    )


def _chat_error(exc: Exception) -> tuple[int, str, str]:
    if isinstance(exc, SessionNotFoundError):
        return 404, "session_not_found", "Session not found"
    if isinstance(exc, InputTooLarge):
        return 413, "input_too_large", "问题过长，无法放入当前上下文预算，请缩短输入"
    if isinstance(exc, ContextBudgetExceeded):
        return 422, "context_budget_exceeded", "上下文超过当前预算，无法完成本轮请求"
    if isinstance(
        exc, (CompactionError, ModelResponseError, OpenAIError, ToolCallLimitExceeded)
    ):
        return 502, "model_error", "模型处理未正常完成，请稍后重试"
    return 500, "chat_failed", "本轮回答未正常完成"


class ChatStreamingResponse(StreamingResponse):
    """在响应发送结束或断开时关闭会话生成器，包括首帧发送失败。"""

    def __init__(
        self, events: AsyncGenerator[ChatStreamEvent, None], first: ChatStartedEvent
    ):
        self._events = events
        self._first = first
        super().__init__(
            self._body(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async def _body(self):
        yield f"event: {self._first.type}\ndata: {self._first.model_dump_json()}\n\n"
        try:
            async for event in self._events:
                yield f"event: {event.type}\ndata: {event.model_dump_json()}\n\n"
        except Exception as exc:
            logging.getLogger(__name__).exception("流式会话失败")
            _, code, message = _chat_error(exc)
            error = ChatErrorEvent(
                code=code,
                message=message,
                session_id=self._first.session_id,
                turn_id=self._first.turn_id,
            )
            yield f"event: error\ndata: {error.model_dump_json()}\n\n"

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            with CancelScope(shield=True):
                try:
                    await self.body_iterator.aclose()
                finally:
                    await self._events.aclose()


@router.post(
    "/chat/stream",
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}}},
)
async def chat_stream(
    input: ChatRequest,
    service: ConversationServiceDependency,
    context: AgentContextDependency,
    dependencies: ToolDependenciesDependency,
) -> StreamingResponse:
    events = service.chat_stream(input.session_id, input.message, context, dependencies)
    try:
        first = await anext(events)
        if not isinstance(first, ChatStartedEvent):
            raise TypeError("会话缺少开始事件")
    except BaseException as exc:
        with CancelScope(shield=True):
            await events.aclose()
        if isinstance(exc, Exception):
            status_code, _, message = _chat_error(exc)
            if status_code == 500:
                logging.getLogger(__name__).exception("流式会话初始化失败")
            raise HTTPException(status_code=status_code, detail=message) from exc
        raise
    return ChatStreamingResponse(events, first)
