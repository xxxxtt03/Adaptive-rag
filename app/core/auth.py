"""JWT authentication utilities for user identity verification and tenant isolation."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt

from app.core.exceptions import UnauthorizedError


def create_access_token(
    user_id: str,
    secret_key: str,
    *,
    algorithm: str = "HS256",
    expires_delta: timedelta | None = None,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    """Create a signed JWT token for a given user_id."""
    now = datetime.now(timezone.utc)
    expire = now + (expires_delta or timedelta(hours=24))
    payload: dict[str, Any] = {
        "sub": user_id,
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
    }
    if extra_claims:
        payload.update(extra_claims)
    return jwt.encode(payload, secret_key, algorithm=algorithm)


def verify_access_token(
    token: str,
    secret_key: str,
    *,
    algorithm: str = "HS256",
) -> dict[str, Any]:
    """Verify and decode a signed JWT token.

    Raises:
        UnauthorizedError: If the token has expired, is malformed, or missing subject claim.
    """
    try:
        payload = jwt.decode(token, secret_key, algorithms=[algorithm])
    except jwt.ExpiredSignatureError as exc:
        raise UnauthorizedError("token has expired") from exc
    except jwt.PyJWTError as exc:
        raise UnauthorizedError("invalid authentication token") from exc

    if "sub" not in payload or not isinstance(payload["sub"], str):
        raise UnauthorizedError("token missing valid subject claim")

    return payload
