import pytest

from app.core.enums import ToolRisk
from app.tools.base import ToolContext
from app.tools.invoice_tools import IssueInvoiceInput, IssueInvoiceTool


@pytest.mark.asyncio
async def test_issue_personal_invoice_success():
    tool = IssueInvoiceTool()
    assert tool.name == "issue_invoice"
    assert tool.risk == ToolRisk.WRITE

    ctx = ToolContext(user_id="u_demo", session_id="s1", trace_id="t1")
    parsed = IssueInvoiceInput(
        order_id="ORD10001",
        invoice_type="personal",
        title="张三",
        email="zhangsan@example.com",
    )
    result = await tool.execute(parsed, ctx)
    assert result.success is True
    assert result.output["order_id"] == "ORD10001"
    assert result.output["invoice_type"] == "个人普通电子发票"
    assert result.output["title"] == "张三"
    assert "48 小时" in result.output["delivery_estimate"]
    assert "优惠券抵扣部分" in result.output["compliance_notice"]


@pytest.mark.asyncio
async def test_issue_company_vat_invoice_success():
    tool = IssueInvoiceTool()
    ctx = ToolContext(user_id="u_demo", session_id="s1", trace_id="t1")
    parsed = IssueInvoiceInput(
        order_id="ORD10002",
        invoice_type="company_vat",
        title="北京智联创想科技有限公司",
        tax_id="91110108MA01ABCDEF",  # 18-char valid format
        email="finance@example.com",
    )
    result = await tool.execute(parsed, ctx)
    assert result.success is True
    assert result.output["invoice_type"] == "企业增值税专用发票"
    assert result.output["tax_id"] == "91110108MA01ABCDEF"
    assert "5 个工作日" in result.output["delivery_estimate"]


@pytest.mark.asyncio
async def test_issue_company_vat_invoice_invalid_tax_id():
    tool = IssueInvoiceTool()
    ctx = ToolContext(user_id="u_demo", session_id="s1", trace_id="t1")
    parsed = IssueInvoiceInput(
        order_id="ORD10003",
        invoice_type="company_vat",
        title="某某公司",
        tax_id="12345",  # Invalid short tax ID
    )
    result = await tool.execute(parsed, ctx)
    assert result.success is False
    assert "18位统一社会信用代码" in result.output["error"]
