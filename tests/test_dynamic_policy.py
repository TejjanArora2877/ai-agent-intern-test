"""Test verifying that changing policy values in markdown changes responses dynamically without code changes."""

import tempfile
from pathlib import Path
import pytest
from src.rag.parser import KnowledgeBaseParser
from src.rag.bm25_retriever import InMemoryBM25Retriever
from src.agent.core import SupportAgent


def test_offline_response_adapts_to_edited_policy_value(tmp_path):
    """
    Verify that if a policy document changes a policy value (e.g. 30 days -> 60 days),
    the offline agent response changes dynamically to 60 days without modifying Python code.
    """
    # Create temporary knowledge base with 60 days policy
    kb_dir = tmp_path / "knowledge-base"
    kb_dir.mkdir()

    doc_content = """---
document_id: RET-TEST-60
title: Returns Policy
status: active
effective_date: 2026-09-01
policy_authority: official
audience: customer
---

# Returns Policy

## Standard return window

Customers on the standard plan may request a return within **60 calendar days of delivery**.
"""
    (kb_dir / "01-returns-policy-test.md").write_text(doc_content, encoding="utf-8")

    # Instantiate retriever pointing to the modified knowledge base
    custom_retriever = InMemoryBM25Retriever(kb_dir=kb_dir)
    agent = SupportAgent(retriever=custom_retriever, force_mock_mode=True)

    response = agent.respond("How long does a regular customer have to return an item?")

    # The offline response must dynamically include '60 calendar days' from the document
    assert "60 calendar days" in response.answer
    assert "30 calendar days" not in response.answer
    assert len(response.sources) > 0
    assert response.sources[0].file == "01-returns-policy-test.md"
