"""Custom tool for cross-border duty and tax calculation."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field

from app.core.enums import ToolRisk
from app.schemas.agent import ToolResult
from app.tools.base import Tool, ToolContext


class CalculateDutyInput(BaseModel):
    """Input model for cross-border duty calculation."""

    amount: float = Field(..., gt=0, description="商品售价（人民币，元）")
    is_luxury_cosmetics: bool = Field(
        default=False,
        description="是否为高档化妆品（单价>=10元/ml或15元/片）",
    )


class CalculateDutyTool(Tool):
    """Calculates cross-border e-commerce comprehensive duty and tax."""

    name = "calculate_cross_border_tax"
    description = (
        "计算跨境保税仓商品的综合税额。输入商品价格与类目，返回适用的综合税率、税费金额及年度额度扣减提示。"
    )
    risk = ToolRisk.READ  # 只读查询与计算工具，无需人工确认
    input_model = CalculateDutyInput

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        parsed: CalculateDutyInput = self.parse(arguments)  # Pydantic 自动类型校验
        price = parsed.amount

        # 单次限额判断
        if price > 5000.0:
            return ToolResult(
                name=self.name,
                success=False,
                error="SINGLE_LIMIT_EXCEEDED",
                output={
                    "message": f"单次商品金额 ¥{price:.2f} 超过海关规定的单笔 5000 元限额，需按一般贸易全额征税。",
                },
                risk=self.risk,
            )

        # 确定综合税率
        if parsed.is_luxury_cosmetics:
            tax_rate = 0.2305  # 23.05%
            category_desc = "高档化妆品（含消费税折算）"
        else:
            tax_rate = 0.091  # 9.1%
            category_desc = "普通跨境保税商品"

        tax_amount = round(price * tax_rate, 2)
        total_payment = round(price + tax_amount, 2)

        return ToolResult(
            name=self.name,
            success=True,
            output={
                "item_price": price,
                "category": category_desc,
                "tax_rate_percentage": f"{tax_rate * 100:.2f}%",
                "estimated_tax": tax_amount,
                "total_payment": total_payment,
                "annual_quota_deduction": price,
                "notice": "个人年度跨境交易总限额为 ¥26000，下单将扣减对应额度。",
            },
            risk=self.risk,
        )
