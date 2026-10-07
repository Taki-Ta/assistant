from fastapi import APIRouter, HTTPException, status
from openai import OpenAIError

from interview_ai.agent.context.budget import (
    CompactionError,
    ContextBudgetExceeded,
    InputTooLarge,
)
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
from ..schemas import ChatRequest, ChatResponse, SourceResponse

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
