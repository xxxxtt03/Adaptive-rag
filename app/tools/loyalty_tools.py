"""Loyalty points tools: profile query, tier benefits and coupon redemption."""

from __future__ import annotations

import uuid
from typing import ClassVar

from pydantic import BaseModel, Field

from app.core.enums import ToolRisk
from app.schemas.agent import ToolResult
from app.tools.base import Tool, ToolContext

# Mock in-memory user loyalty ledger
_USER_LOYALTY_DB: dict[str, dict] = {
    "u_demo": {
        "tier": "gold",
        "tier_name": "金牌会员",
        "points_balance": 5200,
        "points_expiring_soon": 600,
        "expiry_date": "2026-12-31",
    }
}

# Catalog of redeemable vouchers
_CATALOG = {
    "CPN_10": {
        "name": "10元全场通用券",
        "base_points": 1000,
        "threshold_yuan": 99,
        "min_tier": "normal",
    },
    "CPN_50": {
        "name": "50元无门槛立减券",
        "base_points": 4500,
        "threshold_yuan": 0,
        "min_tier": "normal",
    },
    "CPN_SHIP": {
        "name": "免运费专享券",
        "base_points": 500,
        "threshold_yuan": 0,
        "min_tier": "normal",
    },
    "CPN_100": {
        "name": "100元大额尊享券",
        "base_points": 8000,
        "threshold_yuan": 300,
        "min_tier": "gold",
    },
}

_TIER_DISCOUNT = {
    "normal": 1.0,
    "silver": 0.95,
    "gold": 0.90,
    "diamond": 0.80,
}


class LoyaltyProfileInput(BaseModel):
    user_id: str | None = Field(
        default=None,
        max_length=64,
        description="用户 ID，如 u_demo；省略时默认取当前登录会话用户",
    )


class GetLoyaltyProfileTool(Tool):
    name = "get_loyalty_profile"
    description = (
        "查询用户的会员积分余额、即将过期积分数额、会员等级与当前可兑换的优惠券列表。"
        "何时用：用户询问「我有多少积分」「积分什么时候过期」「我是什么等级会员」「积分能换什么」。"
        "何时不用：只查已有优惠券用 get_coupon；查会员升级与通用权益说明用 get_vip_benefits。"
        "参数 user_id：可选用户标识，默认从会话上下文读取。"
        "返回 {user_id, tier, tier_name, points_balance, points_expiring_soon, available_coupons}。"
    )
    risk = ToolRisk.READ
    input_model = LoyaltyProfileInput

    async def execute(self, parsed: LoyaltyProfileInput, context: ToolContext) -> ToolResult:
        uid = parsed.user_id or context.user_id or "u_demo"
        profile = _USER_LOYALTY_DB.get(
            uid,
            {
                "tier": "normal",
                "tier_name": "普通会员",
                "points_balance": 1200,
                "points_expiring_soon": 0,
                "expiry_date": "2026-12-31",
            },
        )
        tier = profile["tier"]
        discount = _TIER_DISCOUNT.get(tier, 1.0)

        coupons = []
        for code, item in _CATALOG.items():
            req_points = int(item["base_points"] * discount)
            coupons.append(
                {
                    "code": code,
                    "name": item["name"],
                    "required_points": req_points,
                    "original_points": item["base_points"],
                    "threshold_yuan": item["threshold_yuan"],
                    "can_afford": profile["points_balance"] >= req_points,
                }
            )

        return ToolResult(
            name=self.name,
            success=True,
            output={
                "user_id": uid,
                "tier": tier,
                "tier_name": profile["tier_name"],
                "points_balance": profile["points_balance"],
                "points_expiring_soon": profile["points_expiring_soon"],
                "expiry_date": profile["expiry_date"],
                "vip_points_discount": f"{int(discount * 10)}折" if discount < 1.0 else "无折扣",
                "available_coupons": coupons,
            },
        )


class RedeemPointsInput(BaseModel):
    user_id: str | None = Field(
        default=None,
        max_length=64,
        description="用户 ID，如 u_demo；省略时从会话上下文读取",
    )
    coupon_code: str = Field(
        min_length=3,
        max_length=32,
        description="欲兑换的优惠券代号，如 CPN_10, CPN_50, CPN_SHIP, CPN_100",
    )


class RedeemPointsTool(Tool):
    name = "redeem_points"
    description = (
        "消耗用户会员积分兑换指定的优惠券，扣减积分并生成优惠券兑换码存入卡券包。"
        "何时用：用户明确指令使用积分兑换优惠券（如「用积分帮我兑换50元券」「我要换免邮券」）。"
        "何时不用：只查询余额与可兑换项目优先使用 get_loyalty_profile；查询通用活动券使用 get_coupon。"
        "参数 coupon_code：待兑换卡券代码（CPN_10, CPN_50, CPN_SHIP, CPN_100）；user_id：可选用户ID。"
        "返回 {transaction_id, coupon_code, coupon_name, points_deducted, remaining_points, voucher_code}。"
    )
    risk = ToolRisk.WRITE
    input_model = RedeemPointsInput

    async def execute(self, parsed: RedeemPointsInput, context: ToolContext) -> ToolResult:
        uid = parsed.user_id or context.user_id or "u_demo"
        code = parsed.coupon_code.upper().strip()

        if code not in _CATALOG:
            return ToolResult(
                name=self.name,
                success=False,
                output={
                    "error": f"未知的优惠券代码 '{code}'",
                    "supported_coupons": list(_CATALOG.keys()),
                },
            )

        item = _CATALOG[code]
        profile = _USER_LOYALTY_DB.setdefault(
            uid,
            {
                "tier": "normal",
                "tier_name": "普通会员",
                "points_balance": 1200,
                "points_expiring_soon": 0,
                "expiry_date": "2026-12-31",
            },
        )
        tier = profile["tier"]

        # Check tier privilege for high-value vouchers
        tier_ranks = {"normal": 1, "silver": 2, "gold": 3, "diamond": 4}
        if tier_ranks.get(tier, 1) < tier_ranks.get(item["min_tier"], 1):
            return ToolResult(
                name=self.name,
                success=False,
                output={
                    "error": f"该券仅限 {item['min_tier']} 及以上会员兑换，您当前等级为 {tier}",
                    "current_tier": tier,
                    "required_tier": item["min_tier"],
                },
            )

        discount = _TIER_DISCOUNT.get(tier, 1.0)
        needed_points = int(item["base_points"] * discount)

        if profile["points_balance"] < needed_points:
            return ToolResult(
                name=self.name,
                success=False,
                output={
                    "error": "积分余额不足，兑换失败",
                    "current_balance": profile["points_balance"],
                    "required_points": needed_points,
                    "shortage": needed_points - profile["points_balance"],
                },
            )

        # Deduct points
        profile["points_balance"] -= needed_points
        tx_id = f"TX_PTS_{uuid.uuid4().hex[:10].upper()}"
        voucher_code = f"VCH_{code}_{uuid.uuid4().hex[:6].upper()}"

        return ToolResult(
            name=self.name,
            success=True,
            output={
                "transaction_id": tx_id,
                "user_id": uid,
                "coupon_code": code,
                "coupon_name": item["name"],
                "points_deducted": needed_points,
                "remaining_points": profile["points_balance"],
                "voucher_code": voucher_code,
                "note": "兑换成功！优惠券已存入您的个人卡券包，有效期 30 天，下单时勾选即可抵扣。",
            },
        )
