"""Unit tests for the LocalFastRouter low-latency fast path."""

import pytest

from app.agents.nodes.generate import generate_node
from app.agents.nodes.intent import IntentClassifier
from app.core.enums import IntentLabel, QueryComplexity, RetrievalStrategy
from app.providers.fake import OfflineLLMProvider
from app.retrieval.adaptive import AdaptiveRouter
from app.schemas.agent import IntentResult


@pytest.mark.asyncio
async def test_fast_path_exact_greetings() -> None:
    classifier = IntentClassifier(OfflineLLMProvider())

    for greeting in ["你好", "您好！", "hi", "hello", "在吗？", "早上好"]:
        result = await classifier.classify(greeting)
        assert result.label == IntentLabel.GREETING
        assert "fast_path:exact_greeting" in result.rationale


@pytest.mark.asyncio
async def test_fast_path_does_not_intercept_complex_queries() -> None:
    classifier = IntentClassifier(OfflineLLMProvider())

    # Mixed query: greeting + business intent must NOT trigger pure greeting fast path
    result = await classifier.classify("你好，帮我取消订单 ORD10003")
    assert result.rationale != "fast_path:exact_greeting"
    assert result.label != IntentLabel.GREETING


@pytest.mark.asyncio
async def test_adaptive_router_fast_paths_greetings_to_direct() -> None:
    router = AdaptiveRouter(OfflineLLMProvider())
    intent = IntentResult(
        label=IntentLabel.GREETING,
        confidence=1.0,
        rationale="fast_path:exact_greeting",
        prompt_version="fast_path_v1",
    )
    complexity = await router.route("你好", intent)
    assert complexity.strategy == RetrievalStrategy.DIRECT
    assert complexity.complexity == QueryComplexity.SIMPLE
    assert complexity.rationale == "fast_path:social_direct"


@pytest.mark.asyncio
async def test_generate_node_direct_greeting_responds_instantly() -> None:
    intent = IntentResult(
        label=IntentLabel.GREETING,
        confidence=1.0,
        rationale="fast_path:exact_greeting",
        prompt_version="fast_path_v1",
    )
    state = {
        "user_query": "你好",
        "intent": intent,
        "strategy": RetrievalStrategy.DIRECT,
        "retrieved_chunks": [],
        "tool_results": [],
    }
    result = await generate_node(state, OfflineLLMProvider())
    assert "智能客服" in result["final_answer"]
    assert result["refused"] is False
    assert result["citations"] == []
