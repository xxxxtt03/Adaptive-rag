import pytest

from app.tools.address_tools import ParseAddressTool, RawAddressInput
from app.tools.base import ToolContext


@pytest.mark.asyncio
async def test_parse_standard_address_with_masking() -> None:
    tool = ParseAddressTool()
    ctx = ToolContext(user_id="u_demo", session_id="s_test", trace_id="t_test")
    raw = "北京市海淀区中关村南大街1号友谊宾馆 张三 13812345678"
    res = await tool.execute(RawAddressInput(raw_text=raw), ctx)

    assert res.success is True
    assert res.output["recipient"] == "张三"
    assert res.output["masked_phone"] == "138****5678"
    assert res.output["phone_tail"] == "5678"
    assert res.output["province"] == "北京"
    assert res.output["is_remote"] is False
    assert res.output["surcharge_yuan"] == 0.0


@pytest.mark.asyncio
async def test_parse_remote_address_triggers_surcharge() -> None:
    tool = ParseAddressTool()
    ctx = ToolContext(user_id="u_demo", session_id="s_test", trace_id="t_test")
    raw = "新疆维吾尔自治区乌鲁木齐市天山区光明路88号 买买提 15999998888"
    res = await tool.execute(RawAddressInput(raw_text=raw), ctx)

    assert res.success is True
    assert res.output["recipient"] == "买买提"
    assert res.output["masked_phone"] == "159****8888"
    assert res.output["province"] == "新疆"
    assert res.output["is_remote"] is True
    assert res.output["surcharge_yuan"] == 8.0


@pytest.mark.asyncio
async def test_parse_missing_phone_returns_error() -> None:
    tool = ParseAddressTool()
    ctx = ToolContext(user_id="u_demo", session_id="s_test", trace_id="t_test")
    raw = "北京市海淀区中关村南大街1号 张三 没有电话"
    res = await tool.execute(RawAddressInput(raw_text=raw), ctx)

    assert res.success is False
    assert "未在输入文本中识别到有效的11位中国大陆手机号码" in res.output["error"]
