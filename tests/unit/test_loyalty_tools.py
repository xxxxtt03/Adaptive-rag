import pytest

from app.tools.base import ToolContext
from app.tools.loyalty_tools import (
    GetLoyaltyProfileTool,
    LoyaltyProfileInput,
    RedeemPointsInput,
    RedeemPointsTool,
    _USER_LOYALTY_DB,
)


@pytest.fixture(autouse=True)
def reset_loyalty_db() -> None:
    # Reset u_demo before each test
    _USER_LOYALTY_DB["u_demo"] = {
        "tier": "gold",
        "tier_name": "金牌会员",
        "points_balance": 5200,
        "points_expiring_soon": 600,
        "expiry_date": "2026-12-31",
    }
    _USER_LOYALTY_DB.pop("u_test_insufficient", None)
    _USER_LOYALTY_DB.pop("u_test_normal", None)


@pytest.mark.asyncio
async def test_get_loyalty_profile_demo_user() -> None:
    tool = GetLoyaltyProfileTool()
    ctx = ToolContext(user_id="u_demo", session_id="s_test", trace_id="t_test")
    res = await tool.execute(LoyaltyProfileInput(), ctx)
    assert res.success is True
    assert res.output["tier"] == "gold"
    assert res.output["points_balance"] == 5200
    assert res.output["points_expiring_soon"] == 600
    assert res.output["vip_points_discount"] == "9折"
    coupons = res.output["available_coupons"]
    assert len(coupons) >= 3
    # Check that CPN_50 is discounted for gold members (4500 * 0.9 = 4050)
    cpn_50 = next(c for c in coupons if c["code"] == "CPN_50")
    assert cpn_50["required_points"] == 4050


@pytest.mark.asyncio
async def test_get_loyalty_profile_new_user() -> None:
    tool = GetLoyaltyProfileTool()
    ctx = ToolContext(user_id="u_brand_new", session_id="s_test", trace_id="t_test")
    res = await tool.execute(LoyaltyProfileInput(user_id="u_brand_new"), ctx)
    assert res.success is True
    assert res.output["tier"] == "normal"
    assert res.output["points_balance"] == 1200


@pytest.mark.asyncio
async def test_redeem_points_success() -> None:
    tool = RedeemPointsTool()
    ctx = ToolContext(user_id="u_demo", session_id="s_test", trace_id="t_test")
    # Gold tier 10% discount: 4500 -> 4050
    res = await tool.execute(RedeemPointsInput(user_id="u_demo", coupon_code="CPN_50"), ctx)
    assert res.success is True
    assert res.output["points_deducted"] == 4050
    assert res.output["remaining_points"] == 5200 - 4050
    assert res.output["voucher_code"].startswith("VCH_CPN_50")
    assert "TX_PTS_" in res.output["transaction_id"]


@pytest.mark.asyncio
async def test_redeem_points_insufficient() -> None:
    tool = RedeemPointsTool()
    _USER_LOYALTY_DB["u_test_insufficient"] = {
        "tier": "normal",
        "tier_name": "普通会员",
        "points_balance": 200,
        "points_expiring_soon": 0,
        "expiry_date": "2026-12-31",
    }
    ctx = ToolContext(user_id="u_test_insufficient", session_id="s_test", trace_id="t_test")
    res = await tool.execute(
        RedeemPointsInput(user_id="u_test_insufficient", coupon_code="CPN_10"), ctx
    )
    assert res.success is False
    assert "积分余额不足" in res.output["error"]
    assert res.output["shortage"] == 800


@pytest.mark.asyncio
async def test_redeem_points_invalid_code() -> None:
    tool = RedeemPointsTool()
    ctx = ToolContext(user_id="u_demo", session_id="s_test", trace_id="t_test")
    res = await tool.execute(
        RedeemPointsInput(user_id="u_demo", coupon_code="CPN_UNKNOWN"), ctx
    )
    assert res.success is False
    assert "未知的优惠券代码" in res.output["error"]


@pytest.mark.asyncio
async def test_redeem_points_tier_restriction() -> None:
    tool = RedeemPointsTool()
    _USER_LOYALTY_DB["u_test_normal"] = {
        "tier": "normal",
        "tier_name": "普通会员",
        "points_balance": 10000,
        "points_expiring_soon": 0,
        "expiry_date": "2026-12-31",
    }
    ctx = ToolContext(user_id="u_test_normal", session_id="s_test", trace_id="t_test")
    res = await tool.execute(
        RedeemPointsInput(user_id="u_test_normal", coupon_code="CPN_100"), ctx
    )
    assert res.success is False
    assert "仅限 gold 及以上会员兑换" in res.output["error"]
