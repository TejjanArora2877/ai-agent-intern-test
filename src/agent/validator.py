"""Deterministic post-generation validator enforcing privacy, citation, and handoff invariants."""

import re
from typing import List, Optional, Set
from src.models.schemas import AgentResponse, SourceReference, CustomerSafeOrderView, RetrievedChunk

# Generic patterns for sensitive PII scanning
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
STREET_ADDRESS_PATTERN = re.compile(
    r"\b\d{1,5}\s+[A-Za-z0-9\s]{2,25}\s+(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Drive|Dr|Boulevard|Blvd|Way)\b",
    re.IGNORECASE
)
INTERNAL_NOTE_LEAK = re.compile(r"\b(risk_score|warehouse_note|fraud review|risk score)\b", re.IGNORECASE)


class DeterministicValidator:
    """Post-generation validator running deterministic checks on agent responses."""

    @classmethod
    def validate_and_sanitize(
        cls,
        response: AgentResponse,
        user_query: str,
        order_view: Optional[CustomerSafeOrderView] = None,
        retrieved_chunks: Optional[List[RetrievedChunk]] = None,
    ) -> AgentResponse:
        """Apply deterministic invariants to ensure safety, privacy, correct citations, and handoffs."""
        answer = response.answer
        handoff = response.handoff
        sources = list(response.sources)

        # 1. Defense-in-Depth Privacy & PII Scrubbing
        if EMAIL_PATTERN.search(answer):
            answer = EMAIL_PATTERN.sub("[REDACTED EMAIL]", answer)
            handoff = True
        if STREET_ADDRESS_PATTERN.search(answer):
            answer = STREET_ADDRESS_PATTERN.sub("[REDACTED ADDRESS]", answer)
            handoff = True
        if INTERNAL_NOTE_LEAK.search(answer):
            answer = INTERNAL_NOTE_LEAK.sub("[INTERNAL DATA REDACTED]", answer)
            handoff = True

        # 2. Citation Verification & Sanitization based on active official metadata
        if retrieved_chunks:
            # Map of valid active official files
            active_official_files = {
                c.file_name for c in retrieved_chunks
                if c.metadata.status == "active" and c.metadata.policy_authority == "official"
            }
            valid_sources = [s for s in sources if s.file in active_official_files]
            sources = valid_sources

        # 3. Deterministic Handoff Enforcement Rules
        norm_query = user_query.lower()
        norm_ans = answer.lower()

        # Rule A: Order exception or order not found always requires human handoff
        if order_view:
            if order_view.requires_support_review or order_view.status == "exception":
                handoff = True
            if order_view.status == "not_found":
                handoff = True

        # Rule B: Generic operational action requests & defect reports requiring human review
        generic_action_stems = {"cancel", "refund", "exchange", "claim", "damage", "defect", "broken", "faulty", "adjust", "change address", "update address"}
        if any(stem in norm_query for stem in generic_action_stems):
            if any(term in norm_ans for term in ["human", "specialist", "review", "connecting", "cannot approve", "cannot issue", "cannot be guaranteed"]):
                handoff = True

        # Rule C: Source conflict or insufficient information in response
        if any(term in norm_ans for term in ["conflict", "inconsistent", "one says", "differing official sources"]):
            handoff = True
        if any(term in norm_ans for term in ["insufficient information", "information is insufficient", "do not have information", "unable to confirm"]):
            handoff = True

        # Rule D: PII or prompt extraction attempts
        if any(term in norm_query for term in ["email", "address", "risk score", "internal note", "system prompt", "developer override", "hidden instruction"]):
            if any(term in norm_query for term in ["email", "address", "risk score", "internal note", "system prompt", "developer"]):
                handoff = True

        # 4. Stale ETA Protection on Cancelled/Returned Orders
        if order_view and order_view.status in ("cancelled", "returned"):
            if "arriving" in norm_ans or "estimated to arrive" in norm_ans:
                # Remove stale arrival statements if present
                answer = re.sub(r"(?i)estimated to arrive on [^.]+\.", "", answer).strip()
                answer = re.sub(r"(?i)will arrive on [^.]+\.", "", answer).strip()

        return AgentResponse(
            answer=answer,
            sources=sources,
            handoff=handoff,
            tool_calls=response.tool_calls,
            debug_trace=response.debug_trace,
        )
