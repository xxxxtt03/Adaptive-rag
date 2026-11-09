from app.skills.loader import match_skill


def test_match_skill_by_trigger() -> None:
    skill = match_skill("这件衣服能退货吗", "faq_policy")
    assert skill is not None
    assert skill["name"] == "return_handling"
    assert "create_return_request" in skill["tools"]


def test_match_skill_by_intent_when_no_trigger() -> None:
    skill = match_skill("查一下 ORD10001", "order_inquiry")
    assert skill is not None
    assert skill["name"] == "order_tracking"


def test_match_skill_cross_border_tax() -> None:
    skill = match_skill("买这瓶香水要交多少关税", "faq_policy")
    assert skill is not None
    assert skill["name"] == "cross_border_tax"
    assert "calculate_cross_border_tax" in skill["tools"]


def test_no_skill_for_chitchat() -> None:
    assert match_skill("今天天气真好", "chitchat") is None
