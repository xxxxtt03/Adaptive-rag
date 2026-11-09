"""Address parsing and privacy sanitization tool for e-commerce CS."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from app.core.enums import ToolRisk
from app.schemas.agent import ToolResult
from app.tools.base import Tool, ToolContext

# Remote regions requiring extra shipping surcharge according to policy
_REMOTE_PROVINCES = {"新疆", "西藏", "青海", "内蒙古", "宁夏", "海南"}

_PHONE_PATTERN = re.compile(r"(?:(?:\+|00)86)?(1[3-9]\d{9})")
_PROVINCE_PATTERN = re.compile(
    r"(北京|上海|天津|重庆|河北|山西|辽宁|吉林|黑龙江|江苏|浙江|安徽|福建|江西|山东|河南|湖北|湖南|广东|海南|四川|贵州|云南|陕西|甘肃|青海|台湾|内蒙古|广西|西藏|宁夏|新疆|香港|澳门)(?:省|市|自治区|特别行政区)?"
)


class RawAddressInput(BaseModel):
    raw_text: str = Field(
        min_length=6,
        max_length=300,
        description="包含收货人姓名、11位手机号和地址的原始文本，如「北京市海淀区中关村南大街1号 张三 13800138000」",
    )


class ParseAddressTool(Tool):
    name = "parse_and_validate_address"
    description = (
        "结构化解析用户提供的收货地址文本，对手机号执行《个保法》脱敏（掩码中间4位），并识别是否属于偏远地区（需补8元运费）。"
        "何时用：用户提供改地址文本（如「帮我改成发到上海市浦东新区张江高科 李四 13912345678」）。"
        "何时不用：只查订单当前物流用 track_shipment；只查订单详情用 get_order。"
        "参数 raw_text：用户输入的收货人、手机号与地址文本。"
        "返回 {recipient, masked_phone, phone_tail, province, is_remote, surcharge_yuan, compliance_note}。"
    )
    risk = ToolRisk.READ
    input_model = RawAddressInput

    async def execute(self, parsed: RawAddressInput, context: ToolContext) -> ToolResult:
        text = parsed.raw_text.strip()

        # 1. Extract phone number & mask for privacy compliance
        phone_match = _PHONE_PATTERN.search(text)
        if not phone_match:
            return ToolResult(
                name=self.name,
                success=False,
                output={
                    "error": "未在输入文本中识别到有效的11位中国大陆手机号码，请补充提供联系电话。",
                    "raw_text": text,
                },
            )

        full_phone = phone_match.group(1)
        masked_phone = f"{full_phone[:3]}****{full_phone[7:]}"
        phone_tail = full_phone[7:]

        # Remove phone from text for further parsing
        clean_text = text.replace(phone_match.group(0), " ").strip()

        # 2. Extract province
        prov_match = _PROVINCE_PATTERN.search(clean_text)
        province = prov_match.group(1) if prov_match else "未知省份"
        is_remote = province in _REMOTE_PROVINCES
        surcharge = 8.0 if is_remote else 0.0

        # 3. Rough recipient extraction (tokens with 2-4 chars not containing address words)
        words = clean_text.split()
        recipient = "客户"
        address_part = clean_text
        for word in words:
            if 2 <= len(word) <= 4 and not any(kw in word for kw in ("省", "市", "区", "县", "街道", "路", "号", "大厦", "小区")):
                recipient = word
                clean_text = clean_text.replace(word, "").strip()
                address_part = clean_text
                break

        return ToolResult(
            name=self.name,
            success=True,
            output={
                "recipient": recipient,
                "masked_phone": masked_phone,
                "phone_tail": phone_tail,
                "province": province,
                "normalized_address": address_part,
                "is_remote": is_remote,
                "surcharge_yuan": surcharge,
                "compliance_note": "符合《个人信息保护法》与平台客服政策：完整手机号已脱敏，会话仅展示脱敏格式与后4位核验码。",
            },
        )
