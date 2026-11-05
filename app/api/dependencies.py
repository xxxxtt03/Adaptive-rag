"""FastAPI dependency providers."""

from fastapi import Depends, Request

from app.core.auth import verify_access_token
from app.core.exceptions import UnauthorizedError
from app.services.container import AppContainer


def get_container(request: Request) -> AppContainer:
    return request.app.state.container


def get_authenticated_user(
    request: Request,
    container: AppContainer = Depends(get_container),
) -> str | None:
    """Extract and cryptographically verify the current user_id from Authorization header.

    If auth_required is True, missing or invalid credentials raise UnauthorizedError.
    If auth_required is False, an invalid credential still raises UnauthorizedError,
    but omitting the header returns None (allowing dev fallback).
    """
    auth_header = request.headers.get("authorization", "").strip()
    if not auth_header:
        if container.settings.auth_required:
            raise UnauthorizedError("missing authorization header")
        return None

    parts = auth_header.split(maxsplit=1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise UnauthorizedError("invalid authorization scheme, Bearer expected")

    token = parts[1]
    secret = container.settings.jwt_secret_key.get_secret_value()
    algorithm = container.settings.jwt_algorithm
    payload = verify_access_token(token, secret, algorithm=algorithm)
    return payload["sub"]
