"""Chat routes: single-shot JSON and SSE streaming."""

from fastapi import APIRouter, Depends, Request
from sse_starlette import JSONServerSentEvent
from sse_starlette.sse import EventSourceResponse

from app.api.dependencies import get_authenticated_user, get_container
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.container import AppContainer

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    request: Request,
    container: AppContainer = Depends(get_container),
    authenticated_user: str | None = Depends(get_authenticated_user),
) -> ChatResponse:
    """Answer one turn and return the full JSON response."""
    if authenticated_user:
        payload = payload.model_copy(update={"user_id": authenticated_user})
    return await container.chat_service.chat(payload, trace_id=request.state.trace_id)


@router.post("/chat/stream")
async def chat_stream(
    payload: ChatRequest,
    request: Request,
    container: AppContainer = Depends(get_container),
    authenticated_user: str | None = Depends(get_authenticated_user),
) -> EventSourceResponse:
    """Answer one turn as an SSE stream of node, token and answer events."""
    if authenticated_user:
        payload = payload.model_copy(update={"user_id": authenticated_user})

    async def events():
        async for item in container.chat_service.stream(payload, trace_id=request.state.trace_id):
            yield JSONServerSentEvent(item)

    return EventSourceResponse(events())
