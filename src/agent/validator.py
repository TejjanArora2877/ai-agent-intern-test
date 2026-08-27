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
        is_conflict: bool = False,
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

        norm_ans = answer.lower()
        norm_query = user_query.lower()

        # 2. Citation Verification & Multi-Source Grounding Alignment
        if retrieved_chunks:
            # Map of valid active official chunks
            active_official_chunks = [
                c for c in retrieved_chunks
                if c.metadata.status == "active" and c.metadata.policy_authority == "official"
            ]
            active_official_files = {c.file_name for c in active_official_chunks}
            valid_sources = [s for s in sources if s.file in active_official_files]
            cited_files = {s.file for s in valid_sources}

            domain_stems = {"warranti", "ship", "return", "damag", "defect", "care", "membership", "gift"}

            # For answers that materially depend on multiple active official documents,
            # ensure all material supporting documents present in retrieved evidence are cited.
            for c in active_official_chunks:
                if c.file_name in cited_files:
                    continue

                # Filter out chunks whose primary policy domain is not discussed in the answer
                title_heading = f"{c.title} {c.heading}".lower()
                chunk_domains = [s for s in domain_stems if s in title_heading]
                if chunk_domains and not any(s in norm_ans for s in chunk_domains):
                    continue

                title_words = [w for w in re.findall(r"\b[a-z0-9]+\b", c.title.lower()) if len(w) > 3]
                title_matched = len(title_words) >= 2 and sum(1 for w in title_words if w in norm_ans) >= len(title_words) * 0.5

                heading_words = [w for w in re.findall(r"\b[a-z0-9]+\b", c.heading.lower()) if len(w) > 3]
                heading_matched = len(heading_words) >= 1 and all(w in norm_ans for w in heading_words)

                chunk_body_words = [w for w in re.findall(r"\b[a-z0-9]{4,}\b", c.content.lower()) if w not in {"this", "that", "with", "from", "have", "been", "must", "will", "when", "item", "items"}]
                matched_body_words = [w for w in set(chunk_body_words) if w in norm_ans]
                content_matched = len(matched_body_words) >= 3 and (len(matched_body_words) / max(len(set(chunk_body_words)), 1)) >= 0.20

                cited_chunks_text = " ".join(f"{sc.title} {sc.content}".lower() for sc in active_official_chunks if sc.file_name in cited_files)
                cross_ref_matched = c.title.lower() in cited_chunks_text

                if (title_matched or heading_matched or (cross_ref_matched and content_matched) or (content_matched and c.score >= 15.0)) and c.score >= 12.0:
                    valid_sources.append(
                        SourceReference(
                            file=c.file_name,
                            heading=c.heading,
                            document_id=c.metadata.document_id,
                        )
                    )
                    cited_files.add(c.file_name)

            sources = valid_sources

        # 3. Deterministic Handoff Enforcement Rules
        # Rule A: Order exception or order not found always requires human handoff
        if order_view:
            if order_view.requires_support_review or order_view.status == "exception":
                handoff = True
            if order_view.status == "not_found":
                handoff = True

        # Rule B: Damage/defect reports & operational action execution requests requiring human review
        is_defect_report = any(stem in norm_query for stem in ["damag", "defect", "broken", "faulty", "wrong item"])
        is_operational_action_request = any(stem in norm_query for stem in ["cancel", "refund", "exchange", "claim", "adjust", "change address", "update address"])
        
        if is_defect_report or is_operational_action_request:
            if any(term in norm_ans for term in ["human", "specialist", "review", "connecting", "cannot approve", "cannot issue", "cannot be guaranteed", "support", "contact"]):
                handoff = True

        # Rule C: Source conflict or insufficient information in response
        if is_conflict:
            handoff = True
        elif any(term in norm_ans for term in ["conflict", "inconsistent", "contradictory", "contradiction", "differing official sources", "safest interim guidance"]):
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
                answer = re.sub(r"(?i)estimated to arrive on [^.]+\.", "", answer).strip()
                answer = re.sub(r"(?i)will arrive on [^.]+\.", "", answer).strip()

        return AgentResponse(
            answer=answer,
            sources=sources,
            handoff=handoff,
            tool_calls=response.tool_calls,
            debug_trace=response.debug_trace,
        )
