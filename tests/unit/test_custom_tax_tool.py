import pytest

from app.core.enums import ToolRisk
from app.tools.base import ToolContext
from app.tools.custom_tax_tools import CalculateDutyTool


@pytest.mark.asyncio
async def test_calculate_duty_general_goods() -> None:
    tool = CalculateDutyTool()
    assert tool.risk == ToolRisk.READ
    assert tool.name == "calculate_cross_border_tax"

    ctx = ToolContext(user_id="u_demo", session_id="s1", trace_id="t1")

    # 1. 测试 300 元普通商品的 9.1% 综合税
    res = await tool.execute({"amount": 300.0, "is_luxury_cosmetics": False}, ctx)
    assert res.success is True
    assert res.output["estimated_tax"] == 27.30
    assert res.output["total_payment"] == 327.30
    assert "9.10%" in res.output["tax_rate_percentage"]


@pytest.mark.asyncio
async def test_calculate_duty_luxury_goods() -> None:
    tool = CalculateDutyTool()
    ctx = ToolContext(user_id="u_demo", session_id="s1", trace_id="t1")

    # 2. 测试 1000 元高档化妆品的 23.05% 综合税
    res = await tool.execute({"amount": 1000.0, "is_luxury_cosmetics": True}, ctx)
    assert res.success is True
    assert res.output["estimated_tax"] == 230.50
    assert res.output["total_payment"] == 1230.50


@pytest.mark.asyncio
async def test_calculate_duty_exceeds_single_limit() -> None:
    tool = CalculateDutyTool()
    ctx = ToolContext(user_id="u_demo", session_id="s1", trace_id="t1")

    # 3. 测试超过单笔 5000 元限制拦截
    res = await tool.execute({"amount": 5500.0}, ctx)
    assert res.success is False
    assert res.error == "SINGLE_LIMIT_EXCEEDED"
