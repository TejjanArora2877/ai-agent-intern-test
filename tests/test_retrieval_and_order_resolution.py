"""Regression tests for active order context isolation and BM25F retrieval relevance."""

import pytest
from src.agent.core import SupportAgent
from src.agent.session import SessionManager
from src.rag.bm25_retriever import InMemoryBM25Retriever


def test_stale_active_order_context_isolation():
    """
    Test that active order context is preserved for genuine follow-ups (e.g. 'Can I return it?'),
    but does NOT trigger order lookups for subsequent general policy, care, or migration questions.
    """
    mgr = SessionManager()
    session_id = "test_stale_order_session"
    agent = SupportAgent(force_mock_mode=True)

    # Turn 1: Explicit order tracking
    r1 = agent.respond("Where is ORD-1007?", session_id=session_id, session_manager=mgr)
    assert len(r1.tool_calls) == 1
    assert r1.tool_calls[0].arguments["order_id"] == "ORD-1007"
    assert r1.debug_trace.order_query_detected is True
    assert r1.debug_trace.order_id_extracted == "ORD-1007"

    # Turn 2: Genuine follow-up using pronoun 'it'
    r2 = agent.respond("Can I return it?", session_id=session_id, session_manager=mgr)
    assert len(r2.tool_calls) == 1
    assert r2.tool_calls[0].arguments["order_id"] == "ORD-1007"
    assert r2.debug_trace.order_query_detected is True

    # Turn 3: Unrelated product care question (must NOT execute order lookup)
    r3 = agent.respond("Can I put the Breeze Tumbler in the dishwasher?", session_id=session_id, session_manager=mgr)
    assert len(r3.tool_calls) == 0
    assert r3.debug_trace.order_query_detected is False
    assert r3.debug_trace.order_tool_result is None

    # Turn 4: Unrelated general return policy question (must NOT execute order lookup)
    r4 = agent.respond("What is the return policy?", session_id=session_id, session_manager=mgr)
    assert len(r4.tool_calls) == 0
    assert r4.debug_trace.order_query_detected is False
    assert r4.debug_trace.order_tool_result is None

    # Turn 5: Unrelated migration policy query (must NOT execute order lookup)
    r5 = agent.respond("The migration note says returns are allowed for 60 days...", session_id=session_id, session_manager=mgr)
    assert len(r5.tool_calls) == 0
    assert r5.debug_trace.order_query_detected is False
    assert r5.debug_trace.order_tool_result is None


def test_return_window_ranks_above_unrelated_sections():
    """
    Test that a return-window query ranks return-policy sections above
    unrelated warranty, shipping, or product care sections.
    """
    retriever = InMemoryBM25Retriever()
    chunks = retriever.retrieve("How long does a regular customer have to return an unused backpack?", top_k=6)
    
    assert len(chunks) > 0
    top_chunk = chunks[0]
    
    # Top chunk must be from returns policy
    assert "return" in top_chunk.file_name.lower() or "return" in top_chunk.title.lower()
    
    # Verify return policy chunks rank above warranty and product care
    return_ranks = [i for i, c in enumerate(chunks) if "return" in c.file_name.lower()]
    warranty_ranks = [i for i, c in enumerate(chunks) if "warranty" in c.file_name.lower()]
    care_ranks = [i for i, c in enumerate(chunks) if "care" in c.file_name.lower()]
    
    assert len(return_ranks) > 0
    if warranty_ranks:
        assert min(return_ranks) < min(warranty_ranks)
    if care_ranks:
        assert min(return_ranks) < min(care_ranks)


def test_shipping_query_ranks_shipping_above_warranty():
    """
    Test that a shipping question ranks shipping sections above unrelated warranty sections.
    """
    retriever = InMemoryBM25Retriever()
    chunks = retriever.retrieve(
        "If my order subtotal is $60 and I have a standard account, do I get free shipping in the US?",
        top_k=6
    )
    
    assert len(chunks) > 0
    top_chunk = chunks[0]
    assert "shipping" in top_chunk.file_name.lower()
    assert "charges" in top_chunk.heading.lower() or "shipping" in top_chunk.heading.lower()
    
    shipping_ranks = [i for i, c in enumerate(chunks) if "shipping" in c.file_name.lower()]
    warranty_ranks = [i for i, c in enumerate(chunks) if "warranty" in c.file_name.lower()]
    
    assert len(shipping_ranks) > 0
    if warranty_ranks:
        assert min(shipping_ranks) < min(warranty_ranks)


def test_order_only_query_retrieves_no_irrelevant_rag_chunks():
    """
    Test that a pure order tracking/status query retrieves no irrelevant RAG evidence.
    """
    agent = SupportAgent(force_mock_mode=True)
    response = agent.respond("Where is ORD-1007?")
    
    # Pure order tracking query should have looked up the order but retrieved 0 RAG chunks
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].arguments["order_id"] == "ORD-1007"
    assert response.debug_trace.retrieved_chunks == []
    assert len(response.sources) == 0
