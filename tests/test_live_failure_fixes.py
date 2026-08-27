"""Regression tests for the live evaluation failure fixes."""

import pytest
from src.agent.core import SupportAgent
from src.agent.session import SessionManager
from src.agent.validator import DeterministicValidator
from src.models.schemas import AgentResponse, SourceReference, RetrievedChunk, DocumentMetadata
from src.rag.conflict import ConflictDetector
from evaluation.runner import evaluate_response, check_concept_present


def test_conflict_handoff_unconditionally_overridden_to_true():
    """
    Test that when is_conflict is True, DeterministicValidator unconditionally overrides
    handoff to True, even if the model returned handoff=False and did not use trigger words.
    """
    raw_response = AgentResponse(
        answer="The first document mentions hand-washing, but another document mentions dishwasher safe.",
        sources=[
            SourceReference(file="11-product-care.md", heading="Breeze Tumbler"),
            SourceReference(file="12-breeze-tumbler-product-card.md", heading="Cleaning"),
        ],
        handoff=False,
    )

    validated = DeterministicValidator.validate_and_sanitize(
        response=raw_response,
        user_query="Can I put the entire Breeze Tumbler in the dishwasher?",
        is_conflict=True,
    )

    assert validated.handoff is True


def test_multi_source_grounding_citations_end_to_end():
    """
    Test that queries requiring multi-source grounding (e.g. damaged final-sale items)
    cite all material supporting authoritative documents.
    """
    agent = SupportAgent(force_mock_mode=True)
    resp = agent.respond("A final-sale bag arrived with a broken zipper yesterday. Am I completely out of luck?")

    cited_files = {s.file for s in resp.sources}
    assert "03-final-sale-and-promotions.md" in cited_files
    assert "04-damaged-or-wrong-items.md" in cited_files
    assert resp.handoff is True


def test_multi_source_citation_aligns_independent_supporting_documents():
    """
    Regression test proving that when an LLM answer materially depends on two active official
    documents but the model's raw output only listed the first summary document,
    DeterministicValidator generically aligns the sources to include both supporting documents.
    """
    chunk1 = RetrievedChunk(
        chunk_id="RET-2026-02#damaged",
        file_name="03-final-sale-and-promotions.md",
        title="Final Sale and Promotional Purchases",
        heading="Damaged or incorrect items",
        heading_hierarchy=["Damaged or incorrect items"],
        content="If an item marked final sale arrives damaged, defective, or incorrect, the final-sale restriction does not block assistance under the Damaged, Defective, or Wrong Items Policy.",
        score=17.5,
        metadata=DocumentMetadata(
            document_id="RET-2026-02",
            title="Final Sale and Promotional Purchases",
            status="active",
            policy_authority="official",
            customer_answering=True,
        ),
    )
    chunk2 = RetrievedChunk(
        chunk_id="OPS-2026-04#final-sale",
        file_name="04-damaged-or-wrong-items.md",
        title="Damaged, Defective, or Wrong Items",
        heading="Final-sale items",
        heading_hierarchy=["Final-sale items"],
        content="Final-sale items are still eligible for review when they arrive damaged, defective, or incorrect. Photos must be submitted within 7 calendar days.",
        score=17.0,
        metadata=DocumentMetadata(
            document_id="OPS-2026-04",
            title="Damaged, Defective, or Wrong Items",
            status="active",
            policy_authority="official",
            customer_answering=True,
        ),
    )

    # Simulated LLM response that only listed the first document
    raw_llm_response = AgentResponse(
        answer="No, you are not out of luck. Even though marked final sale, defective items with a broken zipper remain eligible for assistance. You must submit photos of the damaged item within 7 calendar days of delivery for review.",
        sources=[SourceReference(file="03-final-sale-and-promotions.md", heading="Damaged or incorrect items")],
        handoff=False,
    )

    validated = DeterministicValidator.validate_and_sanitize(
        response=raw_llm_response,
        user_query="A final-sale bag arrived with a broken zipper yesterday. Am I completely out of luck?",
        retrieved_chunks=[chunk1, chunk2],
        is_conflict=False,
    )

    final_cited_files = {s.file for s in validated.sources}
    assert "03-final-sale-and-promotions.md" in final_cited_files
    assert "04-damaged-or-wrong-items.md" in final_cited_files
    assert validated.handoff is True  # Defect report requires handoff


def test_evaluator_sources_documents_conflict_synonyms():
    """
    Regression test proving that:
    'Our official documents provide conflicting guidance...'
    satisfies the concept:
    'current official sources conflict'
    """
    sentence = "Our official documents provide conflicting guidance regarding cleaning the Breeze Tumbler."
    concept = "current official sources conflict"

    assert check_concept_present(concept, sentence) is True

    # Also test complete evaluate_response
    mock_resp = AgentResponse(
        answer="Our official documents provide conflicting guidance regarding cleaning the Breeze Tumbler. The body must be hand-washed, but the product card states all components are dishwasher safe. As safest interim guidance, we recommend hand-washing the body until confirmed by a human specialist.",
        sources=[
            SourceReference(file="11-product-care.md", heading="Breeze Tumbler"),
            SourceReference(file="12-breeze-tumbler-product-card.md", heading="Cleaning"),
        ],
        handoff=True,
    )

    passed, failures = evaluate_response(
        mock_resp,
        {
            "must_include_concepts": [
                "current official sources conflict",
                "one says hand-wash the body",
                "one says all components are dishwasher safe",
                "human confirmation or safest interim guidance",
            ],
            "required_sources": ["11-product-care.md", "12-breeze-tumbler-product-card.md"],
            "must_not_silently_choose_one": True,
            "tool": "not_called",
            "handoff": True,
        },
    )

    assert passed is True, f"Conflict evaluation failed: {failures}"


def test_international_shipping_completeness_includes_duties():
    """
    Test that international shipping inquiries contain destination, delivery estimate,
    and duties/taxes terms without omitting material conditions.
    """
    agent = SupportAgent(force_mock_mode=True)
    mgr = SessionManager()
    session_id = "test_intl_completeness"

    # Turn 1
    agent.respond("Do you ship internationally?", session_id=session_id, session_manager=mgr)
    # Turn 2
    resp2 = agent.respond("What about Canada, and how long does it take?", session_id=session_id, session_manager=mgr)

    answer_lower = resp2.answer.lower()
    assert "canada" in answer_lower
    assert any(term in answer_lower for term in ["5-9", "5–9", "5 to 9"])
    assert any(term in answer_lower for term in ["duties", "duty", "taxes", "tax", "customs"])
    assert resp2.handoff is False


def test_policy_inquiry_vs_operational_action_handoff():
    """
    Test that policy eligibility questions keep handoff=False while
    operational action requests set handoff=True.
    """
    agent = SupportAgent(force_mock_mode=True)
    mgr = SessionManager()
    session_id = "test_policy_vs_action"

    # Turn 1: Lookup order
    agent.respond("What items are in order ORD-1009 and has it arrived?", session_id=session_id, session_manager=mgr)

    # Turn 2: Policy eligibility question (must NOT handoff)
    r_policy = agent.respond("Can I return the Ridge Daypack because I don't like the red color?", session_id=session_id, session_manager=mgr)
    assert r_policy.handoff is False
    assert "Ridge Daypack" in r_policy.answer or "ridge daypack" in r_policy.answer.lower()
    assert "final sale" in r_policy.answer.lower()

    # Operational action request (must handoff)
    r_action = agent.respond("Can you change the shipping address on ORD-1008 to 456 Elm Street?")
    assert r_action.handoff is True


def test_evaluator_numerical_duration_and_support_synonyms():
    """
    Test that evaluator accepts duration variants (45-day, 45-calendar-day, 45 calendar days)
    and support synonyms (customer support can review).
    """
    mock_resp_45_day = AgentResponse(
        answer="Active TrailPlus members receive a 45-day return window from delivery for eligible items.",
        sources=[SourceReference(file="09-trailplus-membership.md", heading="Return window")],
        handoff=False,
    )
    passed, failures = evaluate_response(
        mock_resp_45_day,
        {
            "must_include": ["45 calendar days", "delivery"],
            "required_sources": ["09-trailplus-membership.md"],
            "handoff": False,
        }
    )
    assert passed is True, f"Failed on duration variant: {failures}"

    assert check_concept_present(
        "human review before approval",
        "customer support can review photos and determine an appropriate remedy"
    ) is True
