import base64
import binascii
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from zhiyin.agent.models import RetrievedSource
from zhiyin.agent.repository import ConversationRepository
from zhiyin.db.models import ConversationItem, ItemType, Session, Turn

from .conversation import SessionNotFoundError


class InvalidCursorError(ValueError):
    pass


def _decode_cursor(cursor: str) -> tuple[datetime, UUID]:
    try:
        raw = base64.b64decode(cursor, altchars=b"-_", validate=True).decode("ascii")
        timestamp, identifier = raw.split("|", 1)
        updated_at = datetime.fromisoformat(timestamp)
        if updated_at.utcoffset() is None:
            raise ValueError("Missing timezone")
        return updated_at, UUID(identifier)
    except (ValueError, UnicodeError, binascii.Error) as exc:
        raise InvalidCursorError("Invalid session cursor") from exc


def _encode_cursor(session: Session) -> str:
    raw = f"{session.updated_at.isoformat()}|{session.id}".encode("ascii")
    return base64.urlsafe_b64encode(raw).decode("ascii")


@dataclass
class SessionPage:
    items: list[Session]
    next_cursor: str | None


@dataclass
class TurnRecord:
    turn: Turn
    items: list[ConversationItem]


@dataclass
class TurnPage:
    items: list[TurnRecord]
    next_cursor: int | None


@dataclass
class SessionService:
    _db: AsyncSession
    _repository: ConversationRepository

    async def list_sessions(
        self, owner_id: str, *, limit: int = 20, cursor: str | None = None
    ) -> SessionPage:
        before = _decode_cursor(cursor) if cursor is not None else None
        sessions = await self._repository.list_sessions(
            owner_id, limit=limit + 1, before=before
        )
        items = sessions[:limit]
        next_cursor = _encode_cursor(items[-1]) if len(sessions) > limit else None
        return SessionPage(items, next_cursor)

    async def get_session(self, session_id: UUID, owner_id: str) -> Session:
        session = await self._repository.get_session(session_id, owner_id)
        if session is None:
            raise SessionNotFoundError(str(session_id))
        return session

    async def list_turns(
        self,
        session_id: UUID,
        owner_id: str,
        *,
        limit: int = 20,
        cursor: int | None = None,
    ) -> TurnPage:
        await self.get_session(session_id, owner_id)
        turns = await self._repository.list_turns(
            session_id, limit=limit + 1, before_sequence=cursor
        )
        selected = turns[:limit]
        next_cursor = selected[-1].sequence if len(turns) > limit else None
        items = await self._repository.list_display_items(
            [turn.id for turn in selected]
        )
        items = await self._restore_sources(items, owner_id)
        grouped = {turn.id: [] for turn in selected}
        for item in items:
            grouped[item.turn_id].append(item)
        return TurnPage(
            [
                TurnRecord(
                    turn, sorted(grouped[turn.id], key=lambda item: item.sequence)
                )
                for turn in reversed(selected)
            ],
            next_cursor,
        )

    async def _restore_sources(
        self, items: list[ConversationItem], owner_id: str
    ) -> list[ConversationItem]:
        # 旧记录只保存 chunk_id 和 score；批量查询补齐，避免每个来源一次查询。
        missing = {
            UUID(source["chunk_id"])
            for item in items
            if item.item_type == ItemType.FUNCTION_CALL_OUTPUT
            for source in item.payload.get("sources", [])
            if "content" not in source
        }
        chunks = (
            await self._repository.list_source_chunks(list(missing), owner_id)
            if missing
            else []
        )
        lookup = {chunk.id: (chunk, name) for chunk, name in chunks}
        restored = []
        for item in items:
            if item.item_type != ItemType.FUNCTION_CALL_OUTPUT:
                restored.append(item)
                continue
            sources = []
            for source in item.payload.get("sources", []):
                if "content" in source:
                    sources.append(source)
                elif (found := lookup.get(UUID(source["chunk_id"]))) is not None:
                    chunk, name = found
                    sources.append(
                        RetrievedSource(
                            chunk_id=chunk.id,
                            document_id=chunk.document_id,
                            document_name=name,
                            content=chunk.content,
                            score=source["score"],
                            headings=chunk.headings,
                            start_line=chunk.start_line,
                            end_line=chunk.end_line,
                        ).model_dump(mode="json")
                    )
            restored.append(
                item.model_copy(
                    update={"payload": {**item.payload, "sources": sources}}
                )
            )
        return restored

    async def delete_session(self, session_id: UUID, owner_id: str) -> None:
        async with self._db.begin():
            session = await self._repository.get_session(
                session_id, owner_id, for_update=True
            )
            if session is None:
                raise SessionNotFoundError(str(session_id))
            await self._repository.delete_session(session)
