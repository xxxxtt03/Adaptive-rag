"""Unit tests for distributed confirmation storage and cross-worker execution."""

import time
from unittest.mock import AsyncMock

import pytest

from app.core.enums import ToolRisk
from app.core.exceptions import InvalidRequestError
from app.infra.confirmation_store import (
    InMemoryConfirmationStore,
    PendingConfirmation,
    RedisConfirmationStore,
)
from app.schemas.chat import ActionProposal, ChatRequest, ChatResponse
from app.services.chat_service import ChatService


def _proposal(action_id: str = "act_1001") -> ActionProposal:
    return ActionProposal(
        action_id=action_id,
        tool_name="cancel_order",
        arguments={"order_id": "ORD10003"},
        risk=ToolRisk.DESTRUCTIVE,
        summary="确认后将取消订单 ORD10003。",
    )


@pytest.mark.asyncio
async def test_in_memory_store_put_get_delete() -> None:
    store = InMemoryConfirmationStore()
    now = time.time()
    confirmation = PendingConfirmation(
        session_id="sess_1",
        user_id="user_1",
        action=_proposal(),
        expires_at=now + 100,
    )
    await store.put("act_1001", confirmation, ttl_seconds=100)

    # Retrieve
    retrieved = await store.get("act_1001")
    assert retrieved is not None
    assert retrieved.session_id == "sess_1"
    assert retrieved.action.action_id == "act_1001"

    # Delete
    await store.delete("act_1001")
    assert await store.get("act_1001") is None


@pytest.mark.asyncio
async def test_in_memory_store_expires() -> None:
    store = InMemoryConfirmationStore()
    now = time.time()
    expired = PendingConfirmation(
        session_id="sess_1",
        user_id="user_1",
        action=_proposal(),
        expires_at=now - 10,  # in the past
    )
    await store.put("act_expired", expired, ttl_seconds=1)
    assert await store.get("act_expired") is None


@pytest.mark.asyncio
async def test_redis_confirmation_store_serialization() -> None:
    mock_redis = AsyncMock()
    mock_redis.get.return_value = None
    mock_redis.set.return_value = True
    mock_redis.delete.return_value = 1

    store = RedisConfirmationStore(mock_redis)
    confirmation = PendingConfirmation(
        session_id="sess_redis",
        user_id="user_redis",
        action=_proposal("act_redis"),
        expires_at=time.time() + 900,
    )

    # Put
    await store.put("act_redis", confirmation, ttl_seconds=900)
    mock_redis.set.assert_awaited_once()
    call_args = mock_redis.set.call_args
    assert "pending_action:act_redis" == call_args[0][0]
    assert "sess_redis" in call_args[0][1]

    # Get
    mock_redis.get.return_value = call_args[0][1]
    loaded = await store.get("act_redis")
    assert loaded is not None
    assert loaded.session_id == "sess_redis"
    assert loaded.action.tool_name == "cancel_order"

    # Delete
    await store.delete("act_redis")
    mock_redis.delete.assert_awaited_once_with("pending_action:act_redis")


@pytest.mark.asyncio
async def test_multi_worker_shared_store_and_single_use() -> None:
    """Simulate Worker A generating proposal and Worker B receiving confirmation."""
    shared_store = InMemoryConfirmationStore()

    # Worker A
    worker_a = ChatService.__new__(ChatService)
    worker_a._confirmation_store = shared_store
    worker_a._pending_actions = {}

    req_a = ChatRequest(message="取消订单 ORD10003", user_id="u_alice")
    resp_a = ChatResponse(
        answer="请确认是否取消订单",
        session_id="sess_shared",
        trace_id="trace_a",
        pending_action=_proposal("act_dist_01"),
    )
    await worker_a._remember_pending(req_a, resp_a)

    # Worker B (completely separate instance sharing the same store)
    worker_b = ChatService.__new__(ChatService)
    worker_b._confirmation_store = shared_store
    worker_b._pending_actions = {}

    confirm_req = ChatRequest(
        message="确认",
        user_id="u_alice",
        session_id="sess_shared",
        confirm_action_id="act_dist_01",
    )
    resolved = await worker_b._resolve_confirmation(confirm_req, "sess_shared")
    assert resolved is not None
    assert resolved.action_id == "act_dist_01"

    # Single-use guarantee: second attempt on Worker B (or Worker A) must fail
    with pytest.raises(InvalidRequestError, match="待确认操作不存在或已过期"):
        await worker_b._resolve_confirmation(confirm_req, "sess_shared")
