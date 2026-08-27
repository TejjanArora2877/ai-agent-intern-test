"""Unit tests for the knowledge base parser and BM25 retriever."""

import pytest
from src.rag.parser import KnowledgeBaseParser
from src.rag.bm25_retriever import InMemoryBM25Retriever
from src.config import KNOWLEDGE_BASE_DIR


@pytest.fixture
def retriever():
    return InMemoryBM25Retriever(kb_dir=KNOWLEDGE_BASE_DIR)


def test_parser_extracts_frontmatter_and_headings():
    fpath = KNOWLEDGE_BASE_DIR / "01-returns-policy-current.md"
    chunks = KnowledgeBaseParser.parse_file(fpath)
    
    assert len(chunks) > 0
    first_chunk = chunks[0]
    assert first_chunk.metadata.document_id == "RET-2026-01"
    assert first_chunk.metadata.status == "active"
    assert first_chunk.metadata.policy_authority == "official"
    assert any("Standard return window" in c.heading for c in chunks)


def test_metadata_precedence_filters_superseded_and_draft_docs(retriever):
    """Ensure superseded and draft unapproved documents are strictly excluded from active search."""
    active_files = {c.file_name for c in retriever.active_chunks}
    
    # Active official files must be present
    assert "01-returns-policy-current.md" in active_files
    assert "09-trailplus-membership.md" in active_files
    assert "07-warranty.md" in active_files

    # Superseded and draft docs must NOT be in active chunks
    assert "02-returns-policy-legacy.md" not in active_files
    assert "14-internal-content-migration-notes.md" not in active_files


def test_retrieve_standard_return_window(retriever):
    chunks = retriever.retrieve("How long does a regular customer have to return an unused backpack?")
    assert len(chunks) > 0
    top_file = chunks[0].file_name
    assert top_file == "01-returns-policy-current.md"
    assert any("30 calendar days" in c.content for c in chunks)


def test_retrieve_trailplus_return_window(retriever):
    chunks = retriever.retrieve("My TrailPlus membership was active when I ordered. What is my return window?")
    assert len(chunks) > 0
    found_trailplus = any(c.file_name == "09-trailplus-membership.md" for c in chunks)
    assert found_trailplus


def test_retrieve_international_shipping(retriever):
    chunks = retriever.retrieve("Do you ship to Canada, and how long does it take?")
    assert len(chunks) > 0
    assert any(c.file_name == "06-international-shipping.md" for c in chunks)


def test_retrieve_warranty_info(retriever):
    chunks = retriever.retrieve("Do all Aster & Row products have a lifetime warranty?")
    assert len(chunks) > 0
    assert any(c.file_name == "07-warranty.md" for c in chunks)


def test_retrieve_breeze_tumbler_conflict_sources(retriever):
    """Tumbler query must retrieve both product card and care guide to surface the conflict."""
    chunks = retriever.retrieve("Can I put the Breeze Tumbler in the dishwasher?")
    files = {c.file_name for c in chunks}
    assert "11-product-care.md" in files
    assert "12-breeze-tumbler-product-card.md" in files
