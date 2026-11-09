"""Invoice issuance and VAT tax invoice management tools."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from app.core.enums import ToolRisk
from app.core.ids import new_id
from app.schemas.agent import ToolResult
from app.tools.base import Tool, ToolContext


class IssueInvoiceInput(BaseModel):
    order_id: str = Field(min_length=6, max_length=32, description="关联的电商订单号，如 ORD10001")
    invoice_type: Literal["personal", "company_vat"] = Field(
        default="personal", description="发票类型：personal(个人电子普票), company_vat(企业增值税专用发票)"
    )
    title: str = Field(min_length=2, max_length=100, description="发票抬头名称（个人姓名或企业全称）")
    tax_id: str | None = Field(
        default=None, description="纳税人识别号 / 统一社会信用代码（企业专票必填，18位字母数字）"
    )
    email: str | None = Field(default=None, description="接收电子发票的电子邮箱")


class IssueInvoiceTool(Tool):
    name = "issue_invoice"
    description = (
        "为已支付或已完成的订单申请开具发票。"
        "支持个人普通电子发票与企业增值税专用发票。"
        "企业专票必须提供公司全称和18位统一社会信用代码（税号）。"
        "优惠券抵扣与运费金额不计入开票额度。"
    )
    risk = ToolRisk.WRITE
    input_model = IssueInvoiceInput

    async def execute(self, parsed: BaseModel, context: ToolContext) -> ToolResult:
        payload = parsed if isinstance(parsed, IssueInvoiceInput) else IssueInvoiceInput(**parsed.model_dump())

        # 校验企业专票税号
        if payload.invoice_type == "company_vat":
            if not payload.tax_id or not re.match(r"^[0-9A-HJ-NPQRTUWXY]{18}$", payload.tax_id.strip()):
                return ToolResult(
                    name=self.name,
                    success=False,
                    output={
                        "error": "开具企业增值税专用发票必须提供有效的18位统一社会信用代码（税号）。",
                        "order_id": payload.order_id,
                    },
                    risk=self.risk,
                )

        invoice_id = new_id("inv_")
        delivery_window = "确认收货后 48 小时内发送至指定邮箱" if payload.invoice_type == "personal" else "确认收货后 5 个工作日内开具（纸质到付寄出）"

        return ToolResult(
            name=self.name,
            success=True,
            output={
                "invoice_id": invoice_id,
                "order_id": payload.order_id,
                "invoice_type": "个人普通电子发票" if payload.invoice_type == "personal" else "企业增值税专用发票",
                "title": payload.title,
                "tax_id": payload.tax_id or "不适用",
                "email": payload.email or "系统默认绑定的联系邮箱",
                "status": "APPLIED",
                "delivery_estimate": delivery_window,
                "compliance_notice": "依据平台发票管理细则，优惠券抵扣部分、运费、礼品卡支付部分不开具发票，仅开具实付商品款。",
            },
            risk=self.risk,
        )
