import hashlib
import math
import struct
import pytest

from app.agents.nodes.prototypical_router import PrototypicalIntentRouter
from app.core.enums import IntentLabel
from app.retrieval.tokenizer import tokenize


class CharacterAwareHashEmbedder:
    """Deterministic token-aware embedding provider for unit tests."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        tokens = tokenize(text) or [text]
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for offset in range(0, min(len(digest), self.dim * 4), 4):
                idx = struct.unpack_from(">I", digest, offset)[0] % self.dim
                vector[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


@pytest.mark.asyncio
async def test_prototypical_router_initialization():
    embedder = CharacterAwareHashEmbedder(dim=128)
    router = PrototypicalIntentRouter(embedder, confidence_threshold=0.5)

    assert not router._initialized
    await router.initialize()
    assert router._initialized
    assert len(router._centroids) == len(IntentLabel)


@pytest.mark.asyncio
async def test_prototypical_router_logistics_match():
    embedder = CharacterAwareHashEmbedder(dim=256)
    router = PrototypicalIntentRouter(embedder, confidence_threshold=0.2)
    await router.initialize()

    # Query matches logistics tokens (快递, 进度)
    result = await router.classify("查一下快递发货进度")
    assert result is not None
    assert isinstance(result.label, IntentLabel)
    assert result.confidence > 0.0
    assert "prototypical_router" in result.rationale


@pytest.mark.asyncio
async def test_prototypical_router_fallback_below_threshold():
    embedder = CharacterAwareHashEmbedder(dim=256)
    # Set impossible threshold to test fallback behavior
    router = PrototypicalIntentRouter(embedder, confidence_threshold=0.999)
    await router.initialize()

    result = await router.classify("完全不相干的罕见文字")
    assert result is None


@pytest.mark.asyncio
async def test_intent_classifier_uses_prototypical_router_fast_path():
    from app.agents.nodes.intent import IntentClassifier
    from app.providers.fake import OfflineLLMProvider

    embedder = CharacterAwareHashEmbedder(dim=256)
    proto_router = PrototypicalIntentRouter(embedder, confidence_threshold=0.2)
    classifier = IntentClassifier(OfflineLLMProvider(), prototypical_router=proto_router)

    # Query matches logistics via prototypical router
    result = await classifier.classify("查一下快递发货进度")
    assert result is not None
    assert isinstance(result.label, IntentLabel)
    assert "prototypical_router" in result.rationale
