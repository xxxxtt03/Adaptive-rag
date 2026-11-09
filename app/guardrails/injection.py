"""Prompt-injection detection."""

from __future__ import annotations

import re

from app.core.error_codes import ErrorCode
from app.core.exceptions import GuardrailError

_INJECTION_PATTERNS = (
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.I),
    re.compile(r"忽略(以上|之前|上面)(的)?(指令|提示|设定|限制)", re.I),
    re.compile(r"you\s+are\s+now\s+", re.I),
    re.compile(r"从现在起(，| )?(你不再是|你是一(?:个|位)|扮演)", re.I),
    re.compile(r"system\s*prompt", re.I),
    re.compile(r"(?:输出|显示|告诉我|复述)(?:你的)?(?:系统提示词|初始设定|开发者指令|前置提示)", re.I),
    re.compile(r"(?:repeat|output|show)\s+(?:all\s+)?(?:above|previous|initial)\s+(?:prompt|instructions)", re.I),
    re.compile(r"exfiltrat|send\s+all\s+(user\s+)?(data|information)", re.I),
    re.compile(r"<(?:\|im_start\||\|im_end\||\|system\||\|user\||\|assistant\|)>", re.I),
    re.compile(r"\[/?INST\]", re.I),
    re.compile(r"(?:进入|开启|切换到)(?:DAN|开发者|调试|越狱)模式", re.I),
)


def detect_prompt_injection(text: str) -> None:
    """User text is untrusted. Retrieved docs are also untrusted — caller decides.

    We block obvious jailbreaks at the query boundary. Subtle cases are handled
    by isolating retrieved content behind UNTRUSTED_CONTEXT_BANNER.
    """
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            raise GuardrailError(
                "query looks like prompt injection",
                error_code=ErrorCode.PROMPT_INJECTION_BLOCKED,
            )
