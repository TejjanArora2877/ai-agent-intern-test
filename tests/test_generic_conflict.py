"""Regression test verifying generic conflict detection on a brand new synthetic entity/document pair."""

from pathlib import Path
import pytest
from src.rag.bm25_retriever import InMemoryBM25Retriever
from src.rag.conflict import ConflictDetector
from src.agent.core import SupportAgent


def test_generic_conflict_detection_on_synthetic_product(tmp_path):
    """
    Test that generic conflict detection detects contradictory care instructions
    for a brand new synthetic product without ANY product/filename conditionals in code.
    """
    kb_dir = tmp_path / "knowledge-base"
    kb_dir.mkdir()

    doc1 = """---
document_id: CARE-SYNTH-01
title: Synthetic Gear Care
status: active
effective_date: 2026-08-01
policy_authority: official
audience: customer
---

# Synthetic Gear Care

## Vortex Jacket Care

The stainless zippers and fabrics of the Vortex Jacket must be hand-washed only. Do not machine wash.
"""

    doc2 = """---
document_id: PROD-SYNTH-02
title: Vortex Jacket Overview
status: active
effective_date: 2026-08-01
policy_authority: official
audience: customer
---

# Vortex Jacket Overview

## Cleaning and Maintenance

All components of the Vortex Jacket are completely machine washable.
"""

    (kb_dir / "90-synthetic-care.md").write_text(doc1, encoding="utf-8")
    (kb_dir / "91-synthetic-product.md").write_text(doc2, encoding="utf-8")

    retriever = InMemoryBM25Retriever(kb_dir=kb_dir)
    agent = SupportAgent(retriever=retriever, force_mock_mode=True)

    response = agent.respond("How do I wash the Vortex Jacket?")

    # Assert generic conflict detection was triggered
    assert response.handoff is True
    assert "conflict" in response.answer.lower()
    
    # Assert both synthetic documents were cited
    cited_files = {s.file for s in response.sources}
    assert "90-synthetic-care.md" in cited_files
    assert "91-synthetic-product.md" in cited_files
