"""Comprehensive integration and security tests for Aster & Row Support Agent Web API."""

import pytest
from fastapi.testclient import TestClient

from src.web.app import app, session_manager


@pytest.fixture
def client():
    """Create a FastAPI TestClient instance."""
    return TestClient(app)


def test_health_endpoint(client):
    """Test health check returns system metadata and default offline mode."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["default_mode"] == "offline"
    assert "model" in data
    assert "live_llm_configured" in data


def test_root_serves_html(client):
    """Test GET / returns the frontend HTML application."""
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "Aster &amp; Row" in res.text or "Aster & Row" in res.text


def test_create_new_session(client):
    """Test session creation generates unique session IDs."""
    res1 = client.post("/api/session/new")
    assert res1.status_code == 200
    data1 = res1.json()
    assert "session_id" in data1
    assert data1["session_id"].startswith("web_")

    res2 = client.post("/api/session/new")
    assert res2.status_code == 200
    data2 = res2.json()
    assert data1["session_id"] != data2["session_id"]


def test_offline_chat_works_without_external_api(client):
    """
    Test that offline chat operates deterministically without requiring
    any external LLM API credentials or network calls.
    """
    res = client.post(
        "/api/chat",
        json={
            "message": "What is the return window for regular customers?",
            "mode": "offline",
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert "session_id" in data
    assert isinstance(data["answer"], str) and len(data["answer"]) > 0
    assert isinstance(data["sources"], list)
    assert isinstance(data["handoff"], bool)
    assert data["debug_trace"] is not None
    assert data["debug_trace"]["model_mode"] == "mock"


def test_raw_order_pii_and_internal_notes_not_in_api_response(client):
    """
    Test that raw PII (emails, shipping addresses) and sensitive internal fields
    (risk score, warehouse notes) NEVER appear in API responses or debug traces.
    """
    res = client.post(
        "/api/chat",
        json={
            "message": "Where is order ORD-1001?",
            "mode": "offline",
        },
    )
    assert res.status_code == 200
    res_str = res.text.lower()

    # Assert forbidden sensitive internal fields never leak in JSON string
    forbidden_tokens = [
        "risk_score",
        "warehouse_note",
        "fraud review",
        "customer_email",
        "shipping_address",
        "john.doe@example.com",
        "123 Maple Street",
    ]
    for token in forbidden_tokens:
        assert token.lower() not in res_str, f"Sensitive token '{token}' found in API response!"


def test_multiturn_session_retains_context(client):
    """
    Test that multi-turn conversations persist across turns using session_id.
    """
    session_res = client.post("/api/session/new")
    session_id = session_res.json()["session_id"]

    # Turn 1: Lookup order ORD-1009
    res1 = client.post(
        "/api/chat",
        json={
            "message": "What items are in order ORD-1009 and has it arrived?",
            "session_id": session_id,
            "mode": "offline",
        },
    )
    assert res1.status_code == 200
    assert "ORD-1009" in res1.json()["answer"]

    # Turn 2: Policy eligibility question referring to items from Turn 1
    res2 = client.post(
        "/api/chat",
        json={
            "message": "Can I return the Ridge Daypack because I don't like the red color?",
            "session_id": session_id,
            "mode": "offline",
        },
    )
    assert res2.status_code == 200
    data2 = res2.json()
    assert "Ridge Daypack" in data2["answer"] or "ridge daypack" in data2["answer"].lower()
    assert "final sale" in data2["answer"].lower()
    assert data2["handoff"] is False


def test_new_session_isolation_does_not_inherit_previous_state(client):
    """
    Test that a new session starts with a clean slate and does not inherit
    the active order context from previous sessions.
    """
    session_a = client.post("/api/session/new").json()["session_id"]
    session_b = client.post("/api/session/new").json()["session_id"]

    # Session A sets active order
    client.post(
        "/api/chat",
        json={
            "message": "Where is ORD-1007?",
            "session_id": session_a,
            "mode": "offline",
        },
    )

    # Session B asks a general policy question
    res_b = client.post(
        "/api/chat",
        json={
            "message": "What is the warranty policy for your products?",
            "session_id": session_b,
            "mode": "offline",
        },
    )
    assert res_b.status_code == 200
    data_b = res_b.json()
    # Ensure Session B did not inherit or lookup ORD-1007
    assert "ORD-1007" not in data_b["answer"]
    assert data_b["debug_trace"]["order_query_detected"] is False
    assert data_b["debug_trace"]["order_id_extracted"] is None


def test_conflict_and_handoff_information_returned_correctly(client):
    """
    Test that conflicting source policies trigger both citations, conflict trace,
    and handoff=True in the API response.
    """
    res = client.post(
        "/api/chat",
        json={
            "message": "Can I put the entire Breeze Tumbler in the dishwasher?",
            "mode": "offline",
        },
    )
    assert res.status_code == 200
    data = res.json()

    assert data["handoff"] is True
    cited_files = {s["file"] for s in data["sources"]}
    assert "11-product-care.md" in cited_files
    assert "12-breeze-tumbler-product-card.md" in cited_files
    assert data["debug_trace"]["conflict_detected"] is True


def test_invalid_and_empty_requests_handled_safely(client):
    """
    Test that empty messages, whitespace-only messages, and invalid payloads
    return 400/422 status codes without crashing the server.
    """
    # 1. Empty message string
    res_empty = client.post("/api/chat", json={"message": ""})
    assert res_empty.status_code in (400, 422)

    # 2. Whitespace-only message string
    res_ws = client.post("/api/chat", json={"message": "    "})
    assert res_ws.status_code in (400, 422)

    # 3. Missing message field
    res_missing = client.post("/api/chat", json={"mode": "offline"})
    assert res_missing.status_code == 422


def test_clear_session_endpoint(client):
    """
    Test that DELETE /api/session/{session_id} clears conversation memory.
    """
    session_id = client.post("/api/session/new").json()["session_id"]

    # Add message
    client.post(
        "/api/chat",
        json={"message": "Hello!", "session_id": session_id, "mode": "offline"},
    )

    # Check history exists
    hist_res = client.get(f"/api/session/{session_id}")
    assert hist_res.status_code == 200
    assert len(hist_res.json()["messages"]) >= 2

    # Clear history
    del_res = client.delete(f"/api/session/{session_id}")
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "cleared"

    # Verify history is now empty
    hist_after = client.get(f"/api/session/{session_id}")
    assert len(hist_after.json()["messages"]) == 0
