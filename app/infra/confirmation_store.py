"""Distributed confirmation storage for Human-in-the-Loop high-risk actions.

Supports multi-worker/multi-container deployments via Redis,
with fallback to in-memory storage for offline / single-process environments.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from app.schemas.chat import ActionProposal

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PendingConfirmation:
    session_id: str
    user_id: str
    action: ActionProposal
    expires_at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "action": self.action.model_dump(mode="json"),
            "expires_at": self.expires_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PendingConfirmation:
        return cls(
            session_id=data["session_id"],
            user_id=data["user_id"],
            action=ActionProposal.model_validate(data["action"]),
            expires_at=float(data["expires_at"]),
        )


class ConfirmationStore(ABC):
    @abstractmethod
    async def get(self, action_id: str) -> PendingConfirmation | None:
        """Retrieve a pending confirmation by action_id. Returns None if absent or expired."""

    @abstractmethod
    async def put(
        self, action_id: str, confirmation: PendingConfirmation, ttl_seconds: int = 900
    ) -> None:
        """Store a pending confirmation with a TTL in seconds."""

    @abstractmethod
    async def delete(self, action_id: str) -> None:
        """Remove a pending confirmation once executed or invalidated."""


class InMemoryConfirmationStore(ConfirmationStore):
    """Process-local confirmation store with lazy cleanup. Used in dev or single-process mode."""

    def __init__(self) -> None:
        self._data: dict[str, PendingConfirmation] = {}
        self._lock = asyncio.Lock()

    async def get(self, action_id: str) -> PendingConfirmation | None:
        now = time.time()
        async with self._lock:
            item = self._data.get(action_id)
            if item is None:
                return None
            if item.expires_at <= now:
                del self._data[action_id]
                return None
            return item

    async def put(
        self, action_id: str, confirmation: PendingConfirmation, ttl_seconds: int = 900
    ) -> None:
        now = time.time()
        async with self._lock:
            expired = [k for k, v in self._data.items() if v.expires_at <= now]
            for k in expired:
                del self._data[k]
            self._data[action_id] = confirmation

    async def delete(self, action_id: str) -> None:
        async with self._lock:
            self._data.pop(action_id, None)


class RedisConfirmationStore(ConfirmationStore):
    """Distributed confirmation storage backed by Redis with native TTL expiration."""

    def __init__(self, redis, prefix: str = "pending_action:") -> None:
        self._redis = redis
        self._prefix = prefix

    def _key(self, action_id: str) -> str:
        return f"{self._prefix}{action_id}"

    async def get(self, action_id: str) -> PendingConfirmation | None:
        try:
            raw = await self._redis.get(self._key(action_id))
            if not raw:
                return None
            data = json.loads(raw)
            return PendingConfirmation.from_dict(data)
        except Exception as exc:
            logger.warning("redis_confirmation_store_get_failed: %s", exc)
            return None

    async def put(
        self, action_id: str, confirmation: PendingConfirmation, ttl_seconds: int = 900
    ) -> None:
        try:
            payload = json.dumps(confirmation.to_dict())
            await self._redis.set(self._key(action_id), payload, ex=ttl_seconds)
        except Exception as exc:
            logger.warning("redis_confirmation_store_put_failed: %s", exc)

    async def delete(self, action_id: str) -> None:
        try:
            await self._redis.delete(self._key(action_id))
        except Exception as exc:
            logger.warning("redis_confirmation_store_delete_failed: %s", exc)
