"""Unit and integration tests for JWT authentication and tenant isolation."""

from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from app.config.settings import Settings, clear_settings_cache
from app.core.auth import create_access_token, verify_access_token
from app.core.exceptions import UnauthorizedError
from app.main import create_app

_TEST_SECRET = "test-secret-key-that-is-at-least-32-chars-long!"


def test_jwt_create_and_verify() -> None:
    token = create_access_token("u_alice", _TEST_SECRET)
    claims = verify_access_token(token, _TEST_SECRET)
    assert claims["sub"] == "u_alice"
    assert "exp" in claims
    assert "iat" in claims


def test_jwt_expired_token_raises() -> None:
    token = create_access_token(
        "u_alice", _TEST_SECRET, expires_delta=timedelta(seconds=-10)
    )
    with pytest.raises(UnauthorizedError, match="token has expired"):
        verify_access_token(token, _TEST_SECRET)


def test_jwt_tampered_signature_raises() -> None:
    token = create_access_token("u_alice", _TEST_SECRET)
    tampered = token[:-4] + "abcd"
    with pytest.raises(UnauthorizedError, match="invalid authentication token"):
        verify_access_token(tampered, _TEST_SECRET)


def test_auth_route_token_issuance_and_chat_override() -> None:
    clear_settings_cache()
    app = create_app()
    with TestClient(app) as client:
        # 1. Issue token for user 'u_alice'
        resp = client.post("/api/v1/auth/token", json={"user_id": "u_alice"})
        assert resp.status_code == 200
        token = resp.json()["access_token"]

        # 2. Call chat with valid token and forged body user_id='victim'
        # Server must override body user_id with 'u_alice'
        chat_resp = client.post(
            "/api/v1/chat",
            json={"message": "你好", "user_id": "victim"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert chat_resp.status_code == 200
        assert chat_resp.json()["answer"]


def test_auth_required_blocks_unauthorized_requests() -> None:
    clear_settings_cache()
    strict_settings = Settings(auth_required=True)
    app = create_app(strict_settings)
    with TestClient(app) as client:
        # Request with no Authorization header -> 401
        resp = client.post("/api/v1/chat", json={"message": "你好", "user_id": "u_anon"})
        assert resp.status_code == 401
        body = resp.json()
        assert body["error_code"] == "UNAUTHORIZED"

        # Request with invalid token -> 401
        bad_resp = client.post(
            "/api/v1/chat",
            json={"message": "你好"},
            headers={"Authorization": "Bearer bad.token.here"},
        )
        assert bad_resp.status_code == 401
        assert bad_resp.json()["error_code"] == "UNAUTHORIZED"
