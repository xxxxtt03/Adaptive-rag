"""Lightweight Graph-RAG module for eCommerce policy knowledge graphs.

Solves the multi-hop policy conflict and exception override dilemma
(e.g., general 7-day return vs. custom product exception vs. quality issue override).
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EntityType(str, Enum):
    CATEGORY = "category"  # 商品品类 (e.g. 定制商品, 生鲜食品, 数码)
    POLICY = "policy"  # 核心政策 (e.g. 7天无理由, 质量问题退换, 保修)
    CONDITION = "condition"  # 触发条件 (e.g. 未拆封, 签收7天内, 质量凭证)
    ACTION = "action"  # 执行动作 (e.g. 全额退款, 买家承担运费, 平台承担运费)
    TIMELINE = "timeline"  # 时效窗口 (e.g. 2小时内, 7天内, 15天内, 1-3工作日)


class RelationType(str, Enum):
    APPLIES_TO = "applies_to"  # 适用
    EXCLUDED_FROM = "excluded_from"  # 明确排除/不支持
    EXCEPTION_ALLOWED = "exception_allowed"  # 特殊例外支持
    REQUIRES_CONDITION = "requires_condition"  # 需满足前提
    HAS_TIMELINE = "has_timeline"  # 具有时效约束
    OVERRULES = "overrules"  # 规则覆盖/优先级更高


class EntityNode(BaseModel):
    id: str
    name: str
    entity_type: EntityType
    synonyms: list[str] = Field(default_factory=list)
    description: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class RelationEdge(BaseModel):
    source_id: str
    target_id: str
    relation: RelationType
    description: str = ""
    condition: str = ""


class GraphConstraint(BaseModel):
    entity_name: str
    relation: RelationType
    target_name: str
    description: str
    condition: str = ""


class GraphRetrievalResult(BaseModel):
    matched_entities: list[str] = Field(default_factory=list)
    constraints: list[GraphConstraint] = Field(default_factory=list)
    structured_context: str = ""


class PolicyKnowledgeGraph:
    """In-memory lightweight entity-relation graph with sub-millisecond traversal."""

    def __init__(self) -> None:
        self._nodes: dict[str, EntityNode] = {}
        self._edges: list[RelationEdge] = []
        self._synonym_map: dict[str, str] = {}
        self._matcher_regex: re.Pattern | None = None

    def add_node(self, node: EntityNode) -> None:
        self._nodes[node.id] = node
        self._synonym_map[node.name.lower()] = node.id
        for syn in node.synonyms:
            self._synonym_map[syn.lower()] = node.id
        self._matcher_regex = None  # Invalidate regex cache

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: RelationType,
        description: str = "",
        condition: str = "",
    ) -> None:
        if source_id not in self._nodes:
            raise ValueError(f"Source node {source_id} does not exist in graph")
        if target_id not in self._nodes:
            raise ValueError(f"Target node {target_id} does not exist in graph")
        edge = RelationEdge(
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            description=description,
            condition=condition,
        )
        self._edges.append(edge)

    def _ensure_matcher(self) -> None:
        if self._matcher_regex is None and self._synonym_map:
            # Sort keys by length descending so longer phrases match first
            sorted_keys = sorted(self._synonym_map.keys(), key=len, reverse=True)
            escaped = [re.escape(k) for k in sorted_keys if len(k) >= 2]
            if escaped:
                self._matcher_regex = re.compile("|".join(escaped), re.IGNORECASE)

    def extract_entities(self, query: str) -> list[str]:
        """Extract matching node IDs from user query using exact synonym mapping."""
        self._ensure_matcher()
        if not self._matcher_regex:
            return []
        matched_node_ids: set[str] = set()
        for match in self._matcher_regex.finditer(query):
            term = match.group().lower()
            if term in self._synonym_map:
                matched_node_ids.add(self._synonym_map[term])
        return list(matched_node_ids)

    def query(self, query: str) -> GraphRetrievalResult:
        """Find entities and 1-hop relation constraints to guide LLM inference."""
        node_ids = self.extract_entities(query)
        if not node_ids:
            return GraphRetrievalResult()

        constraints: list[GraphConstraint] = []
        matched_entity_names = [self._nodes[nid].name for nid in node_ids]

        for edge in self._edges:
            # If either source or target is matched
            if edge.source_id in node_ids or edge.target_id in node_ids:
                source_node = self._nodes[edge.source_id]
                target_node = self._nodes[edge.target_id]
                constraints.append(
                    GraphConstraint(
                        entity_name=source_node.name,
                        relation=edge.relation,
                        target_name=target_node.name,
                        description=edge.description,
                        condition=edge.condition,
                    )
                )

        # Build formatted structured text
        lines: list[str] = []
        if constraints:
            lines.append("【政策知识图谱确定性约束 (Graph-RAG)】:")
            for c in constraints:
                cond_str = f" [前提: {c.condition}]" if c.condition else ""
                desc_str = f" ({c.description})" if c.description else ""
                lines.append(f"- {c.entity_name} -> {c.relation.value} -> {c.target_name}{cond_str}{desc_str}")

        return GraphRetrievalResult(
            matched_entities=matched_entity_names,
            constraints=constraints,
            structured_context="\n".join(lines),
        )


def build_ecommerce_policy_graph() -> PolicyKnowledgeGraph:
    """Pre-populates the graph with core e-commerce policies and exception overrides."""
    graph = PolicyKnowledgeGraph()

    # 1. Category Entities
    graph.add_node(
        EntityNode(
            id="cat_custom",
            name="定制商品",
            entity_type=EntityType.CATEGORY,
            synonyms=["定制", "定做", "刻字", "印字", "马克杯", "定制水杯", "定制服装"],
            description="客户指定个性化图案、文案、尺寸制作的专有商品",
        )
    )
    graph.add_node(
        EntityNode(
            id="cat_fresh",
            name="生鲜易腐食品",
            entity_type=EntityType.CATEGORY,
            synonyms=["生鲜", "水果", "海鲜", "冷冻食品", "鲜奶"],
            description="保质期短、易腐烂变质的食品品类",
        )
    )
    graph.add_node(
        EntityNode(
            id="cat_digital",
            name="已拆封数码产品",
            entity_type=EntityType.CATEGORY,
            synonyms=["数码产品", "手机", "平板", "笔记本", "激活数码"],
            description="已撕毁封条、激活系统或通电使用影响二次销售的数码电子产品",
        )
    )
    graph.add_node(
        EntityNode(
            id="cat_general",
            name="普通标准商品",
            entity_type=EntityType.CATEGORY,
            synonyms=["普通商品", "标品", "常规商品"],
            description="未拆封、未受损、具备完整包装的普通在售商品",
        )
    )

    # 2. Policy Entities
    graph.add_node(
        EntityNode(
            id="pol_7d_return",
            name="7天无理由退货",
            entity_type=EntityType.POLICY,
            synonyms=["七天无理由", "7天无理由退换", "无理由退货"],
            description="签收次日起7个自然日内无理由退换服务",
        )
    )
    graph.add_node(
        EntityNode(
            id="pol_quality_issue",
            name="质量问题售后",
            entity_type=EntityType.POLICY,
            synonyms=["质量问题", "破损", "错发", "漏发", "印错字", "错印"],
            description="商品自身缺陷、运输破损或工艺错误引发的售后保障",
        )
    )
    graph.add_node(
        EntityNode(
            id="pol_custom_modify",
            name="定制文案修改窗口",
            entity_type=EntityType.POLICY,
            synonyms=["修改定制", "改文案", "改字", "修改排版"],
            description="定制商品在进入生产前的最后修改时效",
        )
    )

    # 3. Edges: Regular relations & Overrides
    # 普通商品 -> 适用 7天无理由
    graph.add_edge(
        "cat_general",
        "pol_7d_return",
        RelationType.APPLIES_TO,
        description="运费买家自理，商品需完好且不影响二次销售",
        condition="签收次日起7天内",
    )

    # 定制商品 -> 明确排除 7天无理由
    graph.add_edge(
        "cat_custom",
        "pol_7d_return",
        RelationType.EXCLUDED_FROM,
        description="定做尺寸、印字图案不适用7天无理由退换，退货不支持",
    )

    # 定制商品 -> 例外支持 质量问题售后 (覆盖排除规则!)
    graph.add_edge(
        "cat_custom",
        "pol_quality_issue",
        RelationType.EXCEPTION_ALLOWED,
        description="印刷错字、图案颠倒、运输破损全额退款，运费平台承担",
        condition="签收后15日内提交有效凭证",
    )

    # 生鲜食品 -> 明确排除 7天无理由
    graph.add_edge(
        "cat_fresh",
        "pol_7d_return",
        RelationType.EXCLUDED_FROM,
        description="生鲜易腐类由于保质期特殊性不适用无理由退货",
    )

    # 已拆封数码 -> 明确排除 7天无理由
    graph.add_edge(
        "cat_digital",
        "pol_7d_return",
        RelationType.EXCLUDED_FROM,
        description="激活通电后不可二次销售，排除在无理由退货之外",
    )

    # 定制文案修改时效
    graph.add_edge(
        "cat_custom",
        "pol_custom_modify",
        RelationType.HAS_TIMELINE,
        description="超过2小时即进入工厂生产流程，买家无法更改文案",
        condition="下单后2小时内",
    )

    return graph
