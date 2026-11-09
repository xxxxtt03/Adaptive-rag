"""Distributed tracing and micro-span collector for Adaptive-RAG pipeline.

Supports in-process OpenTelemetry-compatible span tracking, flame-graph
generation, and structured span tree export for Jaeger/OTLP ingestion.
"""

from __future__ import annotations

import contextvars
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import Any

_current_trace: contextvars.ContextVar[TraceTree | None] = contextvars.ContextVar(
    "current_trace", default=None
)


@dataclass
class SpanRecord:
    name: str
    trace_id: str
    span_id: str
    parent_id: str | None = None
    start_time: float = field(default_factory=time.perf_counter)
    end_time: float = 0.0
    duration_ms: float = 0.0
    attributes: dict[str, Any] = field(default_factory=dict)
    status: str = "OK"  # "OK" or "ERROR"
    error_message: str | None = None

    def finish(self, status: str = "OK", error: str | None = None) -> None:
        self.end_time = time.perf_counter()
        self.duration_ms = (self.end_time - self.start_time) * 1000.0
        self.status = status
        self.error_message = error

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "duration_ms": round(self.duration_ms, 3),
            "status": self.status,
            "attributes": self.attributes,
            "error_message": self.error_message,
        }


class TraceTree:
    """Request-scoped span container."""

    def __init__(self, trace_id: str) -> None:
        self.trace_id = trace_id
        self.spans: list[SpanRecord] = []
        self._span_counter = 0
        self._active_stack: list[str] = []

    def _next_span_id(self) -> str:
        self._span_counter += 1
        return f"span-{self._span_counter:04d}"

    def start_span(self, name: str, **attributes: Any) -> SpanRecord:
        span_id = self._next_span_id()
        parent_id = self._active_stack[-1] if self._active_stack else None
        span = SpanRecord(
            name=name,
            trace_id=self.trace_id,
            span_id=span_id,
            parent_id=parent_id,
            attributes=attributes,
        )
        self.spans.append(span)
        self._active_stack.append(span_id)
        return span

    def finish_span(self, span: SpanRecord, status: str = "OK", error: str | None = None) -> None:
        span.finish(status=status, error=error)
        if self._active_stack and self._active_stack[-1] == span.span_id:
            self._active_stack.pop()

    def get_summary(self) -> dict[str, Any]:
        total_duration = sum(s.duration_ms for s in self.spans if s.parent_id is None)
        return {
            "trace_id": self.trace_id,
            "total_spans": len(self.spans),
            "root_duration_ms": round(total_duration, 2),
            "spans": [s.to_dict() for s in self.spans],
        }


def init_trace(trace_id: str) -> TraceTree:
    trace = TraceTree(trace_id)
    _current_trace.set(trace)
    return trace


def get_current_trace() -> TraceTree | None:
    return _current_trace.get()


@contextmanager
def trace_span(name: str, **attributes: Any) -> Iterator[SpanRecord | None]:
    trace = _current_trace.get()
    if not trace:
        yield None
        return

    span = trace.start_span(name, **attributes)
    try:
        yield span
        trace.finish_span(span, status="OK")
    except Exception as exc:
        trace.finish_span(span, status="ERROR", error=str(exc))
        raise


@asynccontextmanager
async def async_trace_span(name: str, **attributes: Any) -> AsyncIterator[SpanRecord | None]:
    trace = _current_trace.get()
    if not trace:
        yield None
        return

    span = trace.start_span(name, **attributes)
    try:
        yield span
        trace.finish_span(span, status="OK")
    except Exception as exc:
        trace.finish_span(span, status="ERROR", error=str(exc))
        raise
