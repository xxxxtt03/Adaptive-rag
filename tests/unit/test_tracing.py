import pytest

from app.observability.tracing import (
    async_trace_span,
    get_current_trace,
    init_trace,
    trace_span,
)


def test_init_trace_and_single_span() -> None:
    trace = init_trace("tr-test-001")
    assert get_current_trace() is trace

    with trace_span("intent_classification", query="退货政策") as span:
        assert span is not None
        assert span.name == "intent_classification"
        assert span.attributes["query"] == "退货政策"

    assert span.duration_ms >= 0.0
    assert span.status == "OK"
    assert len(trace.spans) == 1


def test_nested_spans_hierarchy() -> None:
    trace = init_trace("tr-test-002")

    with trace_span("retrieval.pipeline") as parent:
        with trace_span("retrieval.bm25", top_k=20) as bm25_span:
            pass
        with trace_span("retrieval.dense", top_k=20) as dense_span:
            pass
        with trace_span("retrieval.rerank", candidates=8) as rerank_span:
            pass

    assert len(trace.spans) == 4
    root = trace.spans[0]
    assert root.parent_id is None
    # All sub-spans must point to root
    for sub in trace.spans[1:]:
        assert sub.parent_id == root.span_id


def test_span_error_recording() -> None:
    trace = init_trace("tr-test-003")

    with pytest.raises(ValueError, match="simulated failure"):
        with trace_span("tool.execution", tool="cancel_order"):
            raise ValueError("simulated failure")

    assert len(trace.spans) == 1
    span = trace.spans[0]
    assert span.status == "ERROR"
    assert "simulated failure" in str(span.error_message)


@pytest.mark.asyncio
async def test_async_trace_span() -> None:
    trace = init_trace("tr-test-004")

    async with async_trace_span("llm.generation", model="deepseek-v3") as span:
        assert span is not None
        assert span.attributes["model"] == "deepseek-v3"

    assert span.status == "OK"
    assert span.duration_ms >= 0.0


def test_trace_summary_export() -> None:
    trace = init_trace("tr-test-005")

    with trace_span("root_step"):
        with trace_span("child_step", foo="bar"):
            pass

    summary = trace.get_summary()
    assert summary["trace_id"] == "tr-test-005"
    assert summary["total_spans"] == 2
    assert len(summary["spans"]) == 2
    assert summary["spans"][1]["attributes"] == {"foo": "bar"}
