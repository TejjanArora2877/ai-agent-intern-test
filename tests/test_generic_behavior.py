"""Regression tests verifying zero hardcoding of countries, filenames, and product names."""

from pathlib import Path
import re
import pytest

from src.rag.bm25_retriever import InMemoryBM25Retriever
from src.agent.core import SupportAgent
from src.agent.session import SessionManager


def test_dynamic_country_support_from_kb_without_code_changes(tmp_path):
    """
    Verify that newly introduced supported and unsupported countries are handled
    strictly from KB evidence without any country names or filenames in Python code.
    """
    kb_dir = tmp_path / "knowledge-base"
    kb_dir.mkdir()

    # Create a custom shipping policy with an arbitrary filename and new countries (Norway & Japan)
    custom_doc = """---
document_id: SHIP-SYNTH-99
title: International Logistics Policy
status: active
effective_date: 2026-08-01
policy_authority: official
audience: customer
---

# International Logistics Policy

## Supported Destinations

Aster & Row currently ships internationally only to Norway and Japan. Shipping to other international destinations is not available at this time.
"""
    (kb_dir / "99-custom-logistics.md").write_text(custom_doc, encoding="utf-8")

    retriever = InMemoryBM25Retriever(kb_dir=kb_dir)
    agent = SupportAgent(retriever=retriever, force_mock_mode=True)

    # 1. Test supported country (Norway)
    resp_norway = agent.respond("Can you ship to Norway?")
    assert "not currently available" not in resp_norway.answer.lower()
    assert "norway" in resp_norway.answer.lower()
    cited_files_norway = {s.file for s in resp_norway.sources}
    assert "99-custom-logistics.md" in cited_files_norway

    # 2. Test unsupported country (Brazil)
    resp_brazil = agent.respond("Can you ship an item to Brazil?")
    assert "shipping to brazil is not currently available" in resp_brazil.answer.lower()
    assert "norway and japan" in resp_brazil.answer.lower()
    cited_files_brazil = {s.file for s in resp_brazil.sources}
    assert "99-custom-logistics.md" in cited_files_brazil


def test_new_product_follow_up_without_code_changes():
    """
    Verify that follow-up order inquiries work for any product without having
    the product name in Python code.
    """
    agent = SupportAgent(force_mock_mode=True)
    session_mgr = SessionManager()
    session_id = "test_new_product_session"

    # Turn 1: Lookup order ORD-1002
    resp_1 = agent.respond(
        "What is the status of ORD-1002?",
        session_id=session_id,
        session_manager=session_mgr
    )
    assert resp_1.tool_calls[0].tool_name == "order_lookup"
    assert resp_1.tool_calls[0].arguments["order_id"] == "ORD-1002"

    # Turn 2: Follow-up referencing the item generically without mentioning the order ID
    resp_2 = agent.respond(
        "Can I change the delivery address for this item?",
        session_id=session_id,
        session_manager=session_mgr
    )
    # The agent should associate this with ORD-1002 from session context
    assert len(resp_2.tool_calls) > 0
    assert resp_2.tool_calls[0].tool_name == "order_lookup"
    assert resp_2.tool_calls[0].arguments["order_id"] == "ORD-1002"
    assert "ORD-1002" in resp_2.answer


def test_no_filename_or_country_hardcoding_in_production_code():
    """
    Static code inspection verifying that production files in src/ contain no
    hardcoded KB filenames, country names, or dataset-specific product names.
    """
    project_root = Path(__file__).resolve().parent.parent
    core_code = (project_root / "src" / "agent" / "core.py").read_text(encoding="utf-8")
    validator_code = (project_root / "src" / "agent" / "validator.py").read_text(encoding="utf-8")

    forbidden_terms = [
        # Hardcoded filenames in routing
        "06-international-shipping.md",
        "01-returns-policy-current.md",
        "11-product-care.md",
        "12-breeze-tumbler-product-card.md",
        # Specific country names in core routing
        '"germany"',
        '"france"',
        '"uk"',
        '"mexico"',
        '"japan"',
        '"australia"',
        # Specific product words in session routing
        '"tumbler"',
        '"cube"',
        # Specific test phrases in validator
        "broken zipper",
        "damaged item",
    ]

    for term in forbidden_terms:
        assert term.lower() not in core_code.lower(), f"Forbidden hardcoded term found in core.py: {term}"
        assert term.lower() not in validator_code.lower(), f"Forbidden hardcoded term found in validator.py: {term}"
