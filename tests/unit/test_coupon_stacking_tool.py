import pytest

from app.tools.base import ToolContext
from app.tools.coupon_stacking_tools import (
    CalculateCouponStackingTool,
    CouponStackingInput,
)


@pytest.mark.asyncio
async def test_stacking_new10_and_ship0_success() -> None:
    tool = CalculateCouponStackingTool()
    ctx = ToolContext(user_id="u_demo", session_id="s_test", trace_id="t_test")
    inp = CouponStackingInput(
        order_amount_yuan=120.0,
        coupons_held=["NEW10", "SHIP0"],
        is_seckill=False,
        is_customized=False,
    )
    res = await tool.execute(inp, ctx)

    assert res.success is True
    assert res.output["total_discount_yuan"] == 10.0
    assert res.output["final_payable_yuan"] == 110.0
    assert res.output["shipping_waived"] is True
    assert len(res.output["applicable_coupons"]) == 2
    assert len(res.output["rejected_coupons"]) == 0


@pytest.mark.asyncio
async def test_seckill_rejects_new10_allows_ship0() -> None:
    tool = CalculateCouponStackingTool()
    ctx = ToolContext(user_id="u_demo", session_id="s_test", trace_id="t_test")
    inp = CouponStackingInput(
        order_amount_yuan=150.0,
        coupons_held=["NEW10", "SHIP0"],
        is_seckill=True,
    )
    res = await tool.execute(inp, ctx)

    assert res.success is True
    assert res.output["total_discount_yuan"] == 0.0
    assert res.output["shipping_waived"] is True
    assert any(r["code"] == "NEW10" and "秒杀商品不支持" in r["reason"] for r in res.output["rejected_coupons"])


@pytest.mark.asyncio
async def test_customized_rejects_new10() -> None:
    tool = CalculateCouponStackingTool()
    ctx = ToolContext(user_id="u_demo", session_id="s_test", trace_id="t_test")
    inp = CouponStackingInput(
        order_amount_yuan=100.0,
        coupons_held=["NEW10"],
        is_customized=True,
    )
    res = await tool.execute(inp, ctx)

    assert res.success is True
    assert res.output["total_discount_yuan"] == 0.0
    assert any(r["code"] == "NEW10" and "定制类商品不适用" in r["reason"] for r in res.output["rejected_coupons"])


@pytest.mark.asyncio
async def test_shop50_and_new10_mutual_exclusion_picks_larger() -> None:
    tool = CalculateCouponStackingTool()
    ctx = ToolContext(user_id="u_demo", session_id="s_test", trace_id="t_test")
    inp = CouponStackingInput(
        order_amount_yuan=350.0,
        coupons_held=["NEW10", "SHOP50", "PLUS_SHIP"],
    )
    res = await tool.execute(inp, ctx)

    assert res.success is True
    assert res.output["total_discount_yuan"] == 50.0
    assert res.output["final_payable_yuan"] == 300.0
    assert res.output["shipping_waived"] is True
    # SHOP50 is applied, NEW10 is rejected due to mutual exclusion
    app_codes = [a["code"] for a in res.output["applicable_coupons"]]
    assert "SHOP50" in app_codes
    assert "PLUS_SHIP" in app_codes
    assert any(r["code"] == "NEW10" and "不可叠加" in r["reason"] for r in res.output["rejected_coupons"])
