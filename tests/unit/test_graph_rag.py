import pytest

from app.retrieval.graph_rag import (
    EntityType,
    PolicyKnowledgeGraph,
    RelationType,
    build_ecommerce_policy_graph,
)


def test_ecommerce_policy_graph_initialization():
    graph = build_ecommerce_policy_graph()
    assert "cat_custom" in graph._nodes
    assert "pol_7d_return" in graph._nodes


def test_custom_product_query_extraction_and_overrides():
    graph = build_ecommerce_policy_graph()
    # Query mentions "定制马克杯" and "七天无理由"
    result = graph.query("定制马克杯过了七天能否退货？支持七天无理由吗？")

    assert "定制商品" in result.matched_entities
    assert "7天无理由退货" in result.matched_entities

    relations = {(c.entity_name, c.relation, c.target_name) for c in result.constraints}

    # Verify that EXCLUDED_FROM is captured
    assert ("定制商品", RelationType.EXCLUDED_FROM, "7天无理由退货") in relations

    # Verify that EXCEPTION_ALLOWED is also retrieved to prevent hallucination
    assert ("定制商品", RelationType.EXCEPTION_ALLOWED, "质量问题售后") in relations

    # Check structured context output
    assert "【政策知识图谱确定性约束 (Graph-RAG)】" in result.structured_context
    assert "定制商品 -> excluded_from -> 7天无理由退货" in result.structured_context


def test_fresh_food_query_exclusion():
    graph = build_ecommerce_policy_graph()
    result = graph.query("买的水果生鲜不想要了，支持无理由退款吗？")

    assert "生鲜易腐食品" in result.matched_entities
    assert any(
        c.entity_name == "生鲜易腐食品" and c.relation == RelationType.EXCLUDED_FROM
        for c in result.constraints
    )


def test_unmatched_query_returns_empty():
    graph = build_ecommerce_policy_graph()
    result = graph.query("今天天气怎么样？今天几号了？")

    assert len(result.matched_entities) == 0
    assert len(result.constraints) == 0
    assert result.structured_context == ""
