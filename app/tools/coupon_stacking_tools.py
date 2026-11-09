"""Coupon stacking and best-combination optimization engine."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.core.enums import ToolRisk
from app.schemas.agent import ToolResult
from app.tools.base import Tool, ToolContext


class CouponStackingInput(BaseModel):
    order_amount_yuan: float = Field(
        ge=0.0,
        description="订单商品金额（元），如 120.0 或 350.0",
    )
    coupons_held: list[str] = Field(
        min_length=1,
        description="用户持有的待计算优惠券代号列表，如 ['NEW10', 'SHIP0', 'SHOP50', 'PLUS_SHIP']",
    )
    is_seckill: bool = Field(
        default=False,
        description="是否为限时秒杀商品",
    )
    is_customized: bool = Field(
        default=False,
        description="是否为个性化定制商品",
    )


class CalculateCouponStackingTool(Tool):
    name = "calculate_coupon_stacking"
    description = (
        "核算多张优惠券在特定订单下的可用性、互斥规则、叠加策略与最优省钱组合。"
        "何时用：用户询问多张券能否一起用（如「NEW10 和包邮券能叠加吗」「买秒杀能用券吗」「哪个券最划算」）。"
        "何时不用：只查询用户卡券包列表使用 get_coupon 或 get_loyalty_profile。"
        "参数 order_amount_yuan（订单金额）、coupons_held（待核算券列表）、is_seckill（是否秒杀）、is_customized（是否定制）。"
        "返回 {applicable_coupons, rejected_coupons, total_discount_yuan, final_payable_yuan, best_strategy_summary}。"
    )
    risk = ToolRisk.READ
    input_model = CouponStackingInput

    async def execute(self, parsed: CouponStackingInput, context: ToolContext) -> ToolResult:
        amt = parsed.order_amount_yuan
        held = [c.upper().strip() for c in parsed.coupons_held]

        applicable: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []

        # 1. Evaluate individual eligibility
        has_item_coupons: list[tuple[str, float, str]] = []  # (code, discount, desc)
        has_ship_coupon: tuple[str, str] | None = None

        for code in held:
            if code == "NEW10":
                if parsed.is_seckill:
                    rejected.append({"code": code, "reason": "秒杀商品不支持使用商品满减券 NEW10"})
                elif parsed.is_customized:
                    rejected.append({"code": code, "reason": "定制类商品不适用新客优惠券 NEW10"})
                elif amt < 99.0:
                    rejected.append({"code": code, "reason": f"未达到满 99 元门槛（当前订单金额 {amt:.2f} 元）"})
                else:
                    has_item_coupons.append((code, 10.0, "新客满 99 减 10 元"))

            elif code == "SHOP50":
                if parsed.is_seckill:
                    rejected.append({"code": code, "reason": "秒杀商品不支持使用店铺大促券 SHOP50"})
                elif amt < 300.0:
                    rejected.append({"code": code, "reason": f"未达到满 300 元门槛（当前订单金额 {amt:.2f} 元）"})
                else:
                    has_item_coupons.append((code, 50.0, "大促店铺满 300 减 50 元"))

            elif code in ("SHIP0", "PLUS_SHIP"):
                threshold = 59.0 if code == "SHIP0" else 1.0
                desc = "满 59 免基础运费" if code == "SHIP0" else "PLUS 会员免基础运费券"
                if amt < threshold:
                    rejected.append({"code": code, "reason": f"未达到满 {threshold:.0f} 元免邮门槛"})
                else:
                    # Pick the best shipping voucher if user holds both
                    if has_ship_coupon is None:
                        has_ship_coupon = (code, desc)
                    else:
                        rejected.append({"code": code, "reason": "同单仅支持使用一张运费券，已自动选择最优运费券"})

            else:
                rejected.append({"code": code, "reason": f"未知的优惠券代码 '{code}'"})

        # 2. Item-level mutual exclusivity arbitration (pick biggest discount between NEW10 and SHOP50)
        best_item_coupon: tuple[str, float, str] | None = None
        if has_item_coupons:
            # Sort by discount desc
            has_item_coupons.sort(key=lambda x: x[1], reverse=True)
            best_item_coupon = has_item_coupons[0]
            applicable.append(
                {
                    "code": best_item_coupon[0],
                    "type": "商品立减券",
                    "discount_yuan": best_item_coupon[1],
                    "desc": best_item_coupon[2],
                }
            )
            # Remaining item coupons are mutually exclusive
            for other in has_item_coupons[1:]:
                rejected.append(
                    {
                        "code": other[0],
                        "reason": f"商品立减券不可叠加，系统已优先采用优惠力度更大的 {best_item_coupon[0]}（省 {best_item_coupon[1]:.0f} 元）",
                    }
                )

        # 3. Add stackable shipping coupon
        shipping_waived = False
        if has_ship_coupon is not None:
            shipping_waived = True
            applicable.append(
                {
                    "code": has_ship_coupon[0],
                    "type": "运费抵扣券 (可与商品券叠加)",
                    "discount_yuan": 0.0,  # waived base shipping fee
                    "desc": f"{has_ship_coupon[1]}（免基础运费，偏远附加费除外）",
                }
            )

        total_item_discount = best_item_coupon[1] if best_item_coupon else 0.0
        final_payable = max(0.0, amt - total_item_discount)

        summary_parts = []
        if best_item_coupon:
            summary_parts.append(f"使用 {best_item_coupon[0]} 立减 {best_item_coupon[1]:.0f} 元")
        if shipping_waived:
            summary_parts.append(f"叠加 {has_ship_coupon[0]} 免基础运费")
        if not summary_parts:
            summary_str = "当前订单未能满足所持优惠券的使用条件或受秒杀/定制品类互斥限制。"
        else:
            summary_str = f"最优叠券方案：{' + '.join(summary_parts)}，商品实付金额由 {amt:.2f} 元降至 {final_payable:.2f} 元。"

        return ToolResult(
            name=self.name,
            success=True,
            output={
                "order_amount_yuan": amt,
                "applicable_coupons": applicable,
                "rejected_coupons": rejected,
                "total_discount_yuan": total_item_discount,
                "final_payable_yuan": final_payable,
                "shipping_waived": shipping_waived,
                "best_strategy_summary": summary_str,
            },
        )
