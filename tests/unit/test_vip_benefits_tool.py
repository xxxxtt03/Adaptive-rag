"""Unit tests for GetVipBenefitsTool."""

import pytest

from app.core.enums import ToolRisk
from app.tools.base import ToolContext
from app.tools.knowledge_tools import GetVipBenefitsTool, VipBenefitsInput


@pytest.mark.asyncio
async def test_vip_benefits_tool_tiers() -> None:
    tool = GetVipBenefitsTool()
    assert tool.risk == ToolRisk.READ
    assert tool.name == "get_vip_benefits"

    context = ToolContext(user_id="u_demo", session_id="s_demo", trace_id="t_demo")

    # 1. Gold Tier (Default)
    result_gold = await tool.run({}, context)
    assert result_gold.success is True
    assert result_gold.output["level"] == "gold"
    assert result_gold.output["discount_rate"] == 0.95
    assert result_gold.output["free_shipping"] is True
    assert result_gold.output["exclusive_support"] is True
    assert result_gold.output["points_multiplier"] == 1.5

    # 2. Normal Tier
    result_normal = await tool.run({"level": "normal"}, context)
    assert result_normal.success is True
    assert result_normal.output["discount_rate"] == 1.0
    assert result_normal.output["free_shipping"] is False
    assert result_normal.output["exclusive_support"] is False

    # 3. Silver Tier
    result_silver = await tool.run({"level": "silver"}, context)
    assert result_silver.success is True
    assert result_silver.output["discount_rate"] == 0.98
    assert result_silver.output["points_multiplier"] == 1.2

    # 4. Diamond Tier
    result_diamond = await tool.run({"level": "diamond"}, context)
    assert result_diamond.success is True
    assert result_diamond.output["discount_rate"] == 0.90
    assert result_diamond.output["points_multiplier"] == 2.0

    # 5. Unknown Tier fallback
    result_unknown = await tool.run({"level": "nonexistent_tier"}, context)
    assert result_unknown.success is True
    assert result_unknown.output["discount_rate"] == 1.0
    assert result_unknown.output["free_shipping"] is False
