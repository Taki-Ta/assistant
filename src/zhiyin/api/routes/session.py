from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Response

from zhiyin.agent.errors import SessionNotFoundError
from zhiyin.agent.services.session import InvalidCursorError, TurnRecord
from zhiyin.db.models import ItemType, TurnStatus

from ..dependencies import AgentContextDependency, SessionServiceDependency
from ..schemas import (
    FunctionCallItemResponse,
    FunctionCallOutputItemResponse,
    MessageItemResponse,
    SessionListResponse,
    SessionResponse,
    TurnListResponse,
    TurnResponse,
)

router = APIRouter(prefix="/api/v1/sessions", tags=["session"])
PageLimit = Annotated[int, Query(ge=1, le=100)]


@router.get("")
async def list_sessions(
    service: SessionServiceDependency,
    context: AgentContextDependency,
    limit: PageLimit = 20,
    cursor: Annotated[str | None, Query(max_length=256)] = None,
) -> SessionListResponse:
    try:
        page = await service.list_sessions(context.owner_id, limit=limit, cursor=cursor)
    except InvalidCursorError as exc:
        raise HTTPException(status_code=400, detail="无效的会话分页游标") from exc
    return SessionListResponse(
        items=[SessionResponse.model_validate(item) for item in page.items],
        next_cursor=page.next_cursor,
    )


@router.get("/{session_id}")
async def get_session(
    session_id: UUID,
    service: SessionServiceDependency,
    context: AgentContextDependency,
) -> SessionResponse:
    try:
        session = await service.get_session(session_id, context.owner_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc
    return SessionResponse.model_validate(session)


def _turn_response(record: TurnRecord) -> TurnResponse:
    items = []
    for item in record.items:
        common = {"id": item.id, "sequence": item.sequence}
        if item.item_type == ItemType.MESSAGE:
            items.append(
                MessageItemResponse(
                    **common, role=item.role, text=item.text_content or ""
                )
            )
        elif item.item_type == ItemType.FUNCTION_CALL:
            items.append(
                FunctionCallItemResponse(
                    **common,
                    call_id=item.call_id,
                    tool_name=item.tool_name,
                    arguments=item.payload.get("arguments", {}),
                )
            )
        elif item.item_type == ItemType.FUNCTION_CALL_OUTPUT:
            items.append(
                FunctionCallOutputItemResponse(
                    **common,
                    call_id=item.call_id,
                    tool_name=item.tool_name,
                    output=item.text_content or "",
                    succeeded=item.payload.get("succeeded", False),
                    sources=item.payload.get("sources", []),
                )
            )
    turn = record.turn
    return TurnResponse(
        id=turn.id,
        sequence=turn.sequence,
        status=turn.status,
        created_at=turn.created_at,
        completed_at=turn.completed_at,
        error_message="本轮回答未正常完成"
        if turn.status == TurnStatus.FAILED
        else ("本轮回答已停止" if turn.status == TurnStatus.CANCELLED else None),
        items=items,
    )


@router.get("/{session_id}/turns")
async def list_turns(
    session_id: UUID,
    service: SessionServiceDependency,
    context: AgentContextDependency,
    limit: PageLimit = 20,
    cursor: Annotated[int | None, Query(ge=0, le=2_147_483_647)] = None,
) -> TurnListResponse:
    try:
        page = await service.list_turns(
            session_id, context.owner_id, limit=limit, cursor=cursor
        )
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc
    return TurnListResponse(
        items=[_turn_response(record) for record in page.items],
        next_cursor=page.next_cursor,
    )


@router.delete("/{session_id}", status_code=204)
async def delete_session(
    session_id: UUID,
    service: SessionServiceDependency,
    context: AgentContextDependency,
) -> Response:
    try:
        await service.delete_session(session_id, context.owner_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc
    return Response(status_code=204)
