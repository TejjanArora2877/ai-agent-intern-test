"""Unit tests for the order lookup tool and data security invariants."""

import pytest
from src.tools.order_tool import OrderLookupTool, extract_order_id, normalize_order_id, PROHIBITED_FIELDS


@pytest.fixture
def tool():
    return OrderLookupTool()


def test_order_id_normalization():
    assert normalize_order_id("ORD-1007") == "ORD-1007"
    assert normalize_order_id("ord-1007") == "ORD-1007"
    assert normalize_order_id(" ord-1007. ") == "ORD-1007"
    assert normalize_order_id("Check ORD-1004 please!") == "ORD-1004"
    assert normalize_order_id("") is None
    assert normalize_order_id(None) is None


def test_extract_order_id_from_sentence():
    assert extract_order_id("Where is my order ORD-1007 right now?") == "ORD-1007"
    assert extract_order_id("Can you check ord-1001?") == "ORD-1001"
    assert extract_order_id("No order id here") is None


def test_privacy_invariants_no_pii_or_internal_leak(tool):
    """Ensure sensitive fields are strictly absent from the customer safe view."""
    safe_view = tool.lookup("ORD-1007")
    view_dict = safe_view.model_dump()

    # None of the prohibited field names exist as keys
    for field in PROHIBITED_FIELDS:
        assert field not in view_dict

    # Check serialized content does not contain raw sensitive customer PII
    serialized = str(view_dict)
    assert "ava.morgan@example.test" not in serialized
    assert "220 King Street" not in serialized
    assert "82" not in serialized  # risk score
    assert "fraud review cleared" not in serialized.lower()


def test_cancelled_order_stale_field_suppression(tool):
    """Cancelled orders must not report stale ETA or carrier."""
    safe_view = tool.lookup("ORD-1004")
    assert safe_view.status == "cancelled"
    assert safe_view.carrier is None
    assert safe_view.tracking_number is None
    assert safe_view.estimated_delivery is None


def test_returned_order_stale_field_suppression(tool):
    """Returned orders must not report stale ETA or carrier."""
    safe_view = tool.lookup("ORD-1008")
    assert safe_view.status == "returned"
    assert safe_view.carrier is None
    assert safe_view.tracking_number is None
    assert safe_view.estimated_delivery is None


def test_shipped_without_eta(tool):
    """Shipped order with null ETA must set estimated_delivery_available to False."""
    safe_view = tool.lookup("ORD-1011")
    assert safe_view.status == "shipped"
    assert safe_view.carrier == "Canada Post"
    assert safe_view.estimated_delivery is None
    assert safe_view.estimated_delivery_available is False


def test_exception_order_triggers_handoff(tool):
    """Order with status 'exception' must mandate support review and handoff."""
    safe_view = tool.lookup("ORD-1010")
    assert safe_view.status == "exception"
    assert safe_view.requires_support_review is True
    assert safe_view.handoff_recommended is True


def test_unknown_order_triggers_handoff(tool):
    """Non-existent order must set status 'not_found' and recommend handoff."""
    safe_view = tool.lookup("ORD-9999")
    assert safe_view.status == "not_found"
    assert safe_view.handoff_recommended is True
