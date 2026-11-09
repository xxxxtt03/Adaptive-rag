"""Short-TTL cache for static policy retrieval only.

Never cache order/user-specific tool results. Cache key is query + top_k.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from typing import TYPE_CHECKING

from app.schemas.retrieval import RetrievedChunk

if TYPE_CHECKING:
    from app.providers.base import EmbeddingProvider

logger = logging.getLogger(__name__)


def _looks_like_order_query(query: str) -> bool:
    compact = query.upper().replace(" ", "")
    return "ORD" in compact or "订单" in query


def cache_key(query: str, top_k: int) -> str:
    digest = hashlib.sha256(f"{query}|{top_k}".encode()).hexdigest()[:32]
    return f"ret:{digest}"


class RetrievalCache:
    async def get(self, query: str, top_k: int) -> list[RetrievedChunk] | None:
        return None

    async def set(self, query: str, top_k: int, chunks: list[RetrievedChunk]) -> None:
        return None


class MemoryRetrievalCache(RetrievalCache):
    def __init__(self, *, ttl_seconds: int) -> None:
        self._ttl = ttl_seconds
        self._store: dict[str, tuple[float, list[RetrievedChunk]]] = {}

    async def get(self, query: str, top_k: int) -> list[RetrievedChunk] | None:
        if _looks_like_order_query(query):
            return None
        item = self._store.get(cache_key(query, top_k))
        if item is None:
            return None
        expires_at, chunks = item
        if time.monotonic() > expires_at:
            self._store.pop(cache_key(query, top_k), None)
            return None
        return chunks

    async def set(self, query: str, top_k: int, chunks: list[RetrievedChunk]) -> None:
        if _looks_like_order_query(query):
            return
        self._store[cache_key(query, top_k)] = (time.monotonic() + self._ttl, chunks)


class RedisRetrievalCache(RetrievalCache):
    def __init__(self, redis, *, ttl_seconds: int) -> None:
        self._redis = redis
        self._ttl = ttl_seconds

    async def get(self, query: str, top_k: int) -> list[RetrievedChunk] | None:
        if _looks_like_order_query(query):
            return None
        try:
            raw = await self._redis.get(cache_key(query, top_k))
        except Exception:
            logger.warning("retrieval_cache_redis_get_failed")
            return None
        if not raw:
            return None
        payload = json.loads(raw)
        return [RetrievedChunk.model_validate(item) for item in payload]

    async def set(self, query: str, top_k: int, chunks: list[RetrievedChunk]) -> None:
        if _looks_like_order_query(query):
            return
        try:
            await self._redis.set(
                cache_key(query, top_k),
                json.dumps([c.model_dump(mode="json") for c in chunks], ensure_ascii=False),
                ex=self._ttl,
            )
        except Exception:
            logger.warning("retrieval_cache_redis_set_failed")


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticRetrievalCache(RetrievalCache):
    """Semantic vector similarity cache for retrieval queries.

    Instead of exact string matching, evaluates cosine similarity between
    the incoming query embedding and cached query embeddings.
    If similarity >= threshold (default 0.92), returns the cached chunks.
    """

    def __init__(
        self,
        embedder: EmbeddingProvider,
        *,
        ttl_seconds: int = 3600,
        similarity_threshold: float = 0.92,
        max_entries: int = 500,
    ) -> None:
        self._embedder = embedder
        self._ttl = ttl_seconds
        self._threshold = similarity_threshold
        self._max_entries = max_entries
        # List of (query, vector, top_k, expires_at, chunks)
        self._entries: list[tuple[str, list[float], int, float, list[RetrievedChunk]]] = []

    async def get(self, query: str, top_k: int) -> list[RetrievedChunk] | None:
        if _looks_like_order_query(query):
            return None
        if not self._entries:
            return None

        now = time.monotonic()
        # Clean expired entries
        self._entries = [e for e in self._entries if e[3] > now]
        if not self._entries:
            return None

        # Compute query vector
        query_vec = (await self._embedder.embed_queries([query]))[0]

        best_sim = 0.0
        best_chunks: list[RetrievedChunk] | None = None

        for cached_query, cached_vec, cached_top_k, _, chunks in self._entries:
            if cached_top_k != top_k:
                continue
            sim = _cosine_similarity(query_vec, cached_vec)
            if sim > best_sim:
                best_sim = sim
                if sim >= self._threshold:
                    best_chunks = chunks

        if best_chunks is not None:
            logger.debug(
                "semantic_cache_hit",
                extra={"query": query, "best_similarity": round(best_sim, 4)},
            )
            return best_chunks
        return None

    async def set(self, query: str, top_k: int, chunks: list[RetrievedChunk]) -> None:
        if _looks_like_order_query(query):
            return

        now = time.monotonic()
        query_vec = (await self._embedder.embed_queries([query]))[0]

        # Enforce max entries cap (evict oldest)
        if len(self._entries) >= self._max_entries:
            self._entries.pop(0)

        self._entries.append((query, query_vec, top_k, now + self._ttl, chunks))
