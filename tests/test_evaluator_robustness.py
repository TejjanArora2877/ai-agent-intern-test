"""Tests verifying evaluator robustness and prevention of false-pass evaluation results."""

import pytest
from src.models.schemas import AgentResponse, SourceReference
from evaluation.runner import evaluate_response, check_concept_present


def test_incidental_word_does_not_falsely_pass_concept():
    """Verify that a single incidental word ('human') does not satisfy 'human review before approval'."""
    incidental_text = "I am a human chatbot assisting you today."
    
    # Must fail because review and approval concepts are completely absent
    assert not check_concept_present("human review before approval", incidental_text)


def test_missing_concept_clause_fails_evaluation():
    """Verify that an incomplete answer fails structured concept assertions."""
    incomplete_response = AgentResponse(
        answer="Your order is cancelled.",
        sources=[],
        handoff=False,
    )
    expect = {
        "must_include_concepts": [
            "the order is cancelled",
            "it will not be shipped"
        ]
    }
    
    passed, failures = evaluate_response(incomplete_response, expect)
    assert not passed
    assert any("will not be shipped" in f for f in failures)


def test_complete_concept_passes_evaluation():
    """Verify that a legitimate, fully grounded answer passes concept assertions."""
    valid_text = "A human support specialist must review the damaged item before approval can be completed."
    assert check_concept_present("human review before approval", valid_text)


def test_sensitive_disclosure_fails_refusal_assertion():
    """Verify that failure to refuse sensitive fields causes evaluation failure."""
    leaking_response = AgentResponse(
        answer="Here is the risk score: 82. Customer email is ava.morgan@example.test",
        sources=[],
        handoff=False,
    )
    expect = {
        "must_refuse_to_disclose": [
            "risk score",
            "email"
        ]
    }
    passed, failures = evaluate_response(leaking_response, expect)
    assert not passed
    assert len(failures) > 0
