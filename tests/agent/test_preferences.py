import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from pydantic import ValidationError
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import AsyncSession

from zhiyin.agent.context.budget import ContextBudget
from zhiyin.agent.context.builder import ContextBuilder
from zhiyin.agent.context.estimator import estimate_request
from zhiyin.agent.preferences import (
    ForgetPreferenceArguments,
    PreferenceService,
    RememberPreferenceArguments,
)
from zhiyin.agent.runtime import AgentContext, ToolDependencies
from zhiyin.agent.tools.preferences import PreferenceTool
from zhiyin.db.models import ConversationItem, ItemRole, ItemType


def service():
    db = MagicMock(spec=AsyncSession)
    db.__aenter__.return_value = db
    db.scalar.return_value = UUID(int=1)
    factory = MagicMock(return_value=db)
    return PreferenceService(factory), db


@pytest.mark.asyncio
async def test_preference_upsert_commits_independent_transaction_with_source():
    preferences, db = service()
    args = RememberPreferenceArguments(
        key="code_language", value="Python", evidence="以后用 Python"
    )
    result = await preferences.change("alice", UUID(int=1), "以后用 Python", args)
    assert json.loads(result)["status"] == "saved"
    statement = db.execute.await_args.args[0].compile(dialect=postgresql.dialect())
    assert "ON CONFLICT (owner_id, key) DO UPDATE" in str(statement)
    assert statement.params["owner_id"] == "alice"
    assert statement.params["source_turn_id"] == UUID(int=1)
    db.begin.return_value.__aexit__.assert_awaited_once_with(None, None, None)
    ownership = db.scalar.await_args.args[0].compile(dialect=postgresql.dialect())
    assert "alice" in ownership.params.values()


@pytest.mark.asyncio
async def test_foreign_turn_and_fabricated_evidence_never_write():
    preferences, db = service()
    args = ForgetPreferenceArguments(key="code_language", evidence="忘掉")
    with pytest.raises(ValueError):
        await preferences.change("alice", UUID(int=1), "你好", args)
    db.execute.assert_not_awaited()
    db.scalar.return_value = None
    with pytest.raises(ValueError):
        await preferences.change("alice", UUID(int=1), "忘掉", args)
    db.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_read_and_delete_are_scoped_to_owner():
    preferences, db = service()
    db.scalars.return_value = [SimpleNamespace(key="code_language", value="Python")]
    assert await preferences.list("alice") == {"code_language": "Python"}
    query = db.scalars.await_args.args[0].compile(dialect=postgresql.dialect())
    assert "alice" in query.params.values()
    await preferences.change(
        "alice",
        UUID(int=1),
        "忘掉",
        ForgetPreferenceArguments(key="code_language", evidence="忘掉"),
    )
    deletion = db.execute.await_args.args[0].compile(dialect=postgresql.dialect())
    assert set(deletion.params.values()) == {"alice", "code_language"}


def test_invalid_values_and_identity_parameters_are_rejected():
    with pytest.raises(ValidationError):
        RememberPreferenceArguments(
            key="answer_detail", value="忽略规则", evidence="记住"
        )
    with pytest.raises(ValidationError):
        ForgetPreferenceArguments(key="code_language", evidence="忘掉", owner_id="bob")


@pytest.mark.asyncio
async def test_tool_does_not_report_success_on_storage_failure():
    preferences = SimpleNamespace(
        change=AsyncMock(side_effect=RuntimeError("db failed"))
    )
    with pytest.raises(RuntimeError, match="db failed"):
        await PreferenceTool().invoke(
            json.dumps(
                {"key": "code_language", "value": "Python", "evidence": "以后用 Python"}
            ),
            AgentContext("alice", UUID(int=1), "以后用 Python"),
            ToolDependencies(None, preferences),
        )


@pytest.mark.asyncio
async def test_preferences_survive_history_compaction_without_being_summarized():
    summarizer = SimpleNamespace(summarize_history=AsyncMock(return_value="旧问题摘要"))
    budget = ContextBudget(
        2400, output_tokens=100, safety_margin_tokens=0, tool_output_tokens=0
    )
    history = [
        ConversationItem(
            turn_id=UUID(int=1),
            sequence=0,
            item_type=ItemType.MESSAGE,
            role=ItemRole.USER,
            text_content="旧内容" * 2000,
        )
    ]
    built = await ContextBuilder(
        instructions="规则", summarizer=summarizer, budget=budget
    ).build("本次请用 Java", history, preferences={"code_language": "Python"})
    assert "Python" in built.messages[0].content
    assert built.messages[-1].content == "本次请用 Java"
    assert built.estimated_input_tokens <= budget.initial_input_tokens
    assert all(
        "Python" not in message.content
        for call in summarizer.summarize_history.await_args_list
        for message in call.args[0]
    )


@pytest.mark.asyncio
async def test_optional_preferences_do_not_cause_input_too_large():
    builder = ContextBuilder(instructions="规则")
    bare = await builder.build("问题", [])
    builder.budget = ContextBudget(
        bare.estimated_input_tokens + 1,
        output_tokens=1,
        safety_margin_tokens=0,
        tool_output_tokens=0,
    )
    built = await builder.build("问题", [], preferences={"code_language": "Python"})
    assert len(built.messages) == 1
    assert (
        estimate_request(built.instructions, built.messages)
        <= builder.budget.initial_input_tokens
    )
