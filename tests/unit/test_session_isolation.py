"""Unit tests for cross-user session isolation and anti-IDOR security."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.exceptions import InvalidRequestError
from app.repositories.models import Base
from app.repositories.session_repository import SessionRepository


@pytest.mark.asyncio
async def test_session_isolation_blocks_cross_user_access() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    # 1. User Alice creates session
    async with session_factory() as session:
        repo = SessionRepository(session)
        row = await repo.get_or_create("sess_alice_001", "user_alice")
        assert row.user_id == "user_alice"
        await session.commit()

    # 2. User Alice accesses her own session again -> succeeds
    async with session_factory() as session:
        repo = SessionRepository(session)
        row = await repo.get_or_create("sess_alice_001", "user_alice")
        assert row.user_id == "user_alice"

    # 3. User Bob attempts to access user Alice's session -> raises InvalidRequestError
    async with session_factory() as session:
        repo = SessionRepository(session)
        with pytest.raises(InvalidRequestError) as exc_info:
            await repo.get_or_create("sess_alice_001", "user_bob")
        assert "belongs to another user" in str(exc_info.value)

    await engine.dispose()
