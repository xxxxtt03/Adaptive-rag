"""Authentication routes: token issuance."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.dependencies import get_container
from app.core.auth import create_access_token
from app.services.container import AppContainer

router = APIRouter(prefix="/auth", tags=["auth"])


class TokenRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=64)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


@router.post("/token", response_model=TokenResponse)
async def generate_token(
    payload: TokenRequest,
    container: AppContainer = Depends(get_container),
) -> TokenResponse:
    """Issue a signed JWT access token for a given user identity."""
    token = create_access_token(
        user_id=payload.user_id,
        secret_key=container.settings.jwt_secret_key.get_secret_value(),
        algorithm=container.settings.jwt_algorithm,
    )
    return TokenResponse(access_token=token)
