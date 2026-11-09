from app.tools.factory import build_tool_registry


def test_registry_has_expected_tools() -> None:
    class _Retriever:
        pass

    registry = build_tool_registry(_Retriever())  # type: ignore[arg-type]
    assert len(registry.names()) == 20
    assert "get_order" in registry.names()
    assert "get_vip_benefits" in registry.names()
    assert "calculate_cross_border_tax" in registry.names()
    assert "issue_invoice" in registry.names()
    assert "get_loyalty_profile" in registry.names()
    assert "redeem_points" in registry.names()
    assert "escalate_to_human" in registry.names()
