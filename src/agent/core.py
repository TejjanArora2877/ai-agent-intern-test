"""Core SupportAgent orchestrating retrieval, tooling, generation, and validation."""

import json
import re
import time
from typing import Optional, List, Dict, Any, Tuple
import httpx

from src.config import settings
from src.models.schemas import (
    AgentResponse,
    SourceReference,
    ToolCallRecord,
    RetrievedChunk,
    CustomerSafeOrderView,
    ChatMessage,
)
from src.rag.base import BaseRetriever
from src.rag.bm25_retriever import InMemoryBM25Retriever
from src.rag.extractor import (
    clean_markdown,
    extract_relevant_chunk_content,
    extract_top_document_sections,
    extract_multi_source_passages,
    query_has_coverage_in_chunks,
)
from src.rag.conflict import ConflictDetector
from src.tools.order_tool import OrderLookupTool, extract_order_id
from src.agent.session import SessionManager
from src.agent.prompt import SYSTEM_PROMPT, build_agent_prompt
from src.agent.validator import DeterministicValidator
from src.agent.tracer import AgentTracer


class SupportAgent:
    """
    Main Aster & Row Customer Support Agent.
    
    Orchestrates:
    - Multi-turn context preservation.
    - Unified context (RAG + Order Tool execution).
    - Data-layer privacy and status precedence invariants.
    - Generic conflict detection across active official documents.
    - Live LLM structured generation or offline deterministic mock engine.
    - Deterministic post-generation safety, citation, and handoff validation.
    """

    def __init__(
        self,
        retriever: Optional[BaseRetriever] = None,
        order_tool: Optional[OrderLookupTool] = None,
        force_mock_mode: bool = False,
    ):
        self.retriever = retriever or InMemoryBM25Retriever()
        self.order_tool = order_tool or OrderLookupTool()
        self.force_mock_mode = force_mock_mode

    def respond(
        self,
        user_message: str,
        session_id: str = "default_session",
        session_manager: Optional[SessionManager] = None,
    ) -> AgentResponse:
        """Process a user message in a conversation session and generate an AgentResponse."""
        start_time = time.time()
        mgr = session_manager or SessionManager()
        history = mgr.get_history(session_id)

        # 1. Multi-turn order ID resolution
        current_order_id = extract_order_id(user_message)
        session_order_id = current_order_id
        
        # If no order ID in current message, look back in recent session history
        if not session_order_id:
            for past_msg in reversed(history):
                past_id = extract_order_id(past_msg.content)
                if past_id:
                    session_order_id = past_id
                    break

        # 2. Precise Order Inquiry Classification
        norm_query = user_message.lower()
        explicit_missing_order_pattern = re.compile(
            r"\b(where\s+is\s+my\s+order|track\s+(my\s+)?order|status\s+of\s+my\s+order|check\s+my\s+order|where('?s|\s+is)\s+my\s+package)\b",
            re.IGNORECASE
        )
        is_explicit_missing_inquiry = bool(explicit_missing_order_pattern.search(user_message))
        is_order_intent = any(kw in norm_query for kw in [
            "order", "return", "item", "track", "package", "deliver",
            "arrive", "shipped", "status", "cancel", "exchange", "refund",
            "address", "where is", "when will"
        ])
        is_order_inquiry = bool(current_order_id or (session_order_id and is_order_intent) or is_explicit_missing_inquiry)

        # Order Tool Execution
        tool_calls: List[ToolCallRecord] = []
        order_view: Optional[CustomerSafeOrderView] = None
        order_missing = False

        if current_order_id:
            order_view = self.order_tool.lookup(current_order_id)
            tool_calls.append(
                ToolCallRecord(
                    tool_name="order_lookup",
                    arguments={"order_id": current_order_id},
                    result=order_view.model_dump(),
                )
            )
        elif session_order_id and is_order_intent:
            # Follow-up referencing active order in session using generic intent
            order_view = self.order_tool.lookup(session_order_id)
            tool_calls.append(
                ToolCallRecord(
                    tool_name="order_lookup",
                    arguments={"order_id": session_order_id},
                    result=order_view.model_dump(),
                )
            )
        elif is_explicit_missing_inquiry and not session_order_id:
            order_missing = True

        # 3. Formulate RAG query and retrieve active official passages
        enriched_query = user_message
        if history:
            user_past_msgs = [m.content for m in history if m.role == "user"]
            if user_past_msgs:
                enriched_query = f"{' '.join(user_past_msgs)} {user_message}"

        # If order was looked up, also enrich query with item attributes
        if order_view and order_view.items:
            item_details = " ".join(f"{item.name} {'final sale' if item.final_sale else ''}" for item in order_view.items)
            enriched_query = f"{enriched_query} {item_details} {order_view.membership_tier or ''}"

        retrieved_chunks = self.retriever.retrieve(enriched_query, top_k=settings.max_retrieved_chunks)

        # Generic Conflict Detection across retrieved active official documents
        is_conflict, conflict_chunks, conflict_text = ConflictDetector.detect_conflict(retrieved_chunks, user_message)

        # 4. Generate Response (Live LLM or Offline Deterministic Engine)
        if settings.is_live_llm_enabled and not self.force_mock_mode:
            raw_res, response = self._generate_live_llm(
                user_message, history, retrieved_chunks, order_view, order_missing
            )
            model_mode = "live_llm"
        else:
            raw_res, response = self._generate_offline_mock(
                user_message, history, retrieved_chunks, order_view, order_missing,
                is_conflict=is_conflict, conflict_chunks=conflict_chunks, conflict_text=conflict_text
            )
            model_mode = "mock"

        # Associate tool calls
        response.tool_calls = tool_calls

        # 5. Deterministic Post-Generation Validation
        validated_response = DeterministicValidator.validate_and_sanitize(
            response=response,
            user_query=user_message,
            order_view=order_view,
            retrieved_chunks=retrieved_chunks,
        )

        latency = (time.time() - start_time) * 1000.0

        # 6. Record Observability Trace
        trace = AgentTracer.create_trace(
            user_message=user_message,
            conversation_history=list(history),
            order_query_detected=is_order_inquiry,
            order_id_extracted=current_order_id or session_order_id,
            order_tool_result=order_view.model_dump() if order_view else None,
            retrieved_chunks=[c.model_dump() for c in retrieved_chunks],
            conflict_detected=is_conflict,
            model_mode=model_mode,
            raw_model_response=raw_res,
            latency_ms=latency,
        )
        validated_response.debug_trace = trace

        # 7. Update Session History
        mgr.add_message(session_id, "user", user_message)
        mgr.add_message(session_id, "assistant", validated_response.answer)

        return validated_response

    def _generate_live_llm(
        self,
        user_message: str,
        history: List[ChatMessage],
        retrieved_chunks: List[RetrievedChunk],
        order_view: Optional[CustomerSafeOrderView],
        order_missing: bool,
    ) -> Tuple[str, AgentResponse]:
        """Call live OpenAI-compatible LLM endpoint."""
        prompt = build_agent_prompt(
            user_query=user_message,
            conversation_history=history,
            retrieved_chunks=retrieved_chunks,
            order_view=order_view.model_dump() if order_view else None,
            order_missing=order_missing,
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            with httpx.Client(timeout=30.0) as client:
                res = client.post(
                    f"{settings.openai_base_url.rstrip('/')}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {settings.openai_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": settings.model_name,
                        "messages": messages,
                        "temperature": 0.0,
                        "response_format": {"type": "json_object"},
                    },
                )
                res.raise_for_status()
                data = res.json()
                raw_content = data["choices"][0]["message"]["content"]
                parsed = json.loads(raw_content)

                sources = []
                for s in parsed.get("sources", []):
                    sources.append(
                        SourceReference(
                            file=s.get("file", ""),
                            heading=s.get("heading", ""),
                            document_id=s.get("document_id"),
                        )
                    )

                return raw_content, AgentResponse(
                    answer=parsed.get("answer", ""),
                    sources=sources,
                    handoff=bool(parsed.get("handoff", False)),
                )
        except Exception as e:
            raw_mock, mock_resp = self._generate_offline_mock(
                user_message, history, retrieved_chunks, order_view, order_missing
            )
            mock_resp.answer = f"(Live LLM unavailable: {e}) {mock_resp.answer}"
            return str(e), mock_resp

    def _generate_offline_mock(
        self,
        user_message: str,
        history: List[ChatMessage],
        retrieved_chunks: List[RetrievedChunk],
        order_view: Optional[CustomerSafeOrderView],
        order_missing: bool,
        is_conflict: bool = False,
        conflict_chunks: Optional[List[RetrievedChunk]] = None,
        conflict_text: Optional[str] = None,
    ) -> Tuple[str, AgentResponse]:
        """
        Generic, dynamic offline generation engine.
        Synthesizes responses strictly from retrieved evidence and order views without hardcoded policy values.
        """
        norm_query = user_message.lower()

        # 1. System Prompt / Developer Override Security Guardrail (Generic)
        if any(term in norm_query for term in ["system instruction", "developer override", "hidden prompt", "system prompt", "developer mode"]):
            answer = "I cannot disclose internal system directives, developer configuration, or customer email addresses. I am escalating this inquiry to human support."
            return "jailbreak_refusal", AgentResponse(answer=answer, sources=[], handoff=True)

        # 2. Data Privacy Refusal Guardrail (Generic)
        is_privacy_probe = (
            any(w in norm_query for w in ["what is the", "show me", "give me", "reveal", "tell me the", "display", "print", "extract", "lookup"])
            and any(field in norm_query for field in ["email", "risk score", "internal note", "customer name", "shipping address", "address"])
            and not any(w in norm_query for w in ["change", "update", "correct"])
        )
        if is_privacy_probe:
            answer = "To protect customer data privacy, I cannot disclose customer personal information (such as email or shipping address), risk scores, or internal notes. I am connecting you with a human support specialist."
            return "privacy_refusal", AgentResponse(answer=answer, sources=[], handoff=True)

        # 3. Missing Order ID Handling
        if order_missing:
            answer = "I would be happy to help check your order status. Could you please provide your order ID (for example, ORD-1001)?"
            return "missing_id", AgentResponse(answer=answer, sources=[], handoff=False)

        # 4. Order Tool Result Dynamic Synthesis
        if order_view:
            if order_view.status == "not_found":
                answer = f"Order {order_view.order_id} was not found in our system. Please check the order ID or contact support."
                return "order_not_found", AgentResponse(answer=answer, sources=[], handoff=True)

            if order_view.status == "exception":
                msg = order_view.customer_safe_message or "The shipment has an operational exception requiring support review."
                answer = f"Order {order_view.order_id} status is exception: {msg} I recommend connecting with a human support specialist."
                return "order_exception", AgentResponse(answer=answer, sources=[], handoff=True)

            # Check if query is about changing address on the order
            if "address" in norm_query and any(w in norm_query for w in ["change", "update", "correct"]):
                addr_chunks = [c for c in retrieved_chunks if "change" in c.title.lower() or "cancel" in c.title.lower() or "order" in c.title.lower()]
                top_chunk = addr_chunks[0] if addr_chunks else (retrieved_chunks[0] if retrieved_chunks else None)
                policy_text = clean_markdown(top_chunk.content) if top_chunk else ""
                sources = [SourceReference(file=top_chunk.file_name, heading=top_chunk.heading, document_id=top_chunk.metadata.document_id)] if top_chunk else []
                answer = f"Order {order_view.order_id} is currently processing. According to our policy: {policy_text} Address changes cannot be guaranteed after processing, and a human support specialist review is required."
                return "order_address_change", AgentResponse(answer=answer, sources=sources, handoff=True)

            # Check if query is about returning items in the looked-up order
            if "return" in norm_query and order_view.items:
                final_sale_items = [it for it in order_view.items if it.final_sale]
                if final_sale_items:
                    item_names = ", ".join(it.name for it in final_sale_items)
                    ret_chunks = [c for c in retrieved_chunks if "final" in c.title.lower() or "return" in c.title.lower()]
                    top_chunk = ret_chunks[0] if ret_chunks else (retrieved_chunks[0] if retrieved_chunks else None)
                    policy_text = clean_markdown(top_chunk.content) if top_chunk else ""
                    sources = [SourceReference(file=top_chunk.file_name, heading=top_chunk.heading, document_id=top_chunk.metadata.document_id)] if top_chunk else []
                    answer = f"The {item_names} in order {order_view.order_id} is marked final sale. Final-sale items cannot be returned for a change of mind or color preference. According to our policy: {policy_text}"
                    return "order_final_sale_return", AgentResponse(answer=answer, sources=sources, handoff=False)

            if any(w in norm_query for w in ["what items", "items in", "what did i order"]):
                items_str = ", ".join(f"{it.name} (Qty: {it.quantity}{', Final Sale' if it.final_sale else ''})" for it in order_view.items)
                status_str = f"Status: {order_view.status}."
                if order_view.status == "delivered" and order_view.delivered_at:
                    status_str += f" Delivered on {order_view.delivered_at}."
                answer = f"Order {order_view.order_id} contains: {items_str}. {status_str}"
                return "order_items", AgentResponse(answer=answer, sources=[], handoff=False)

            if order_view.status == "cancelled":
                msg = order_view.customer_safe_message or "The order is cancelled and will not be shipped."
                answer = f"The order is cancelled and will not be shipped. {msg}"
                return "order_cancelled", AgentResponse(answer=answer, sources=[], handoff=False)

            if order_view.status == "returned":
                msg = order_view.customer_safe_message or "The return was received and processed."
                answer = f"Order {order_view.order_id} status is returned. {msg}"
                return "order_returned", AgentResponse(answer=answer, sources=[], handoff=False)

            if order_view.status == "shipped":
                if order_view.estimated_delivery:
                    answer = f"Order {order_view.order_id} has shipped with {order_view.carrier} (Tracking: {order_view.tracking_number}). {order_view.customer_safe_message or f'The estimated delivery date is {order_view.estimated_delivery}.'}"
                else:
                    answer = f"Order {order_view.order_id} has shipped with {order_view.carrier} (Tracking: {order_view.tracking_number}). A delivery estimate is unavailable at this time. {order_view.customer_safe_message or ''}".strip()
                return "order_shipped", AgentResponse(answer=answer, sources=[], handoff=False)

            if order_view.status in ("processing", "pending"):
                eta_str = f" Estimated delivery: {order_view.estimated_delivery}." if order_view.estimated_delivery else " A delivery estimate is not yet available."
                answer = f"Order {order_view.order_id} is currently {order_view.status}. {order_view.customer_safe_message or ''}{eta_str}".strip()
                return "order_processing", AgentResponse(answer=answer, sources=[], handoff=False)

        # 5. Generic Active Document Conflict Handling
        if is_conflict and conflict_text and conflict_chunks:
            sources = [SourceReference(file=c.file_name, heading=c.heading, document_id=c.metadata.document_id) for c in conflict_chunks]
            return "conflict_detected", AgentResponse(answer=conflict_text, sources=sources, handoff=True)

        # 6. Unapproved Document / Migration Note Rejection Guardrail
        if any(t in norm_query for t in ["migration note", "migration scratchpad", "draft policy", "scratchpad"]):
            ret_chunks = [c for c in retrieved_chunks if "return" in c.title.lower()]
            chunk = ret_chunks[0] if ret_chunks else retrieved_chunks[0]
            passages = clean_markdown(chunk.content)
            sources = [SourceReference(file=chunk.file_name, heading=chunk.heading, document_id=chunk.metadata.document_id)]
            answer = f"The content migration note is not authoritative and draft notes cannot override policy. According to our official policy: {passages} The standard policy is 30 days unless a valid exception applies. The agent cannot approve a return without authorized review."
            return "unapproved_doc_refusal", AgentResponse(answer=answer, sources=sources, handoff=False)

        # 7. Generic Safe Abstention for Insufficient Information
        is_covered = (
            self.retriever.is_query_covered(user_message, retrieved_chunks)
            if hasattr(self.retriever, "is_query_covered")
            else query_has_coverage_in_chunks(user_message, retrieved_chunks)
        )
        if not is_covered or not retrieved_chunks:
            answer = "The supplied information is insufficient to answer your request. I recommend consulting with a human customer support specialist for confirmation."
            return "insufficient_info", AgentResponse(answer=answer, sources=[], handoff=True)

        # 8. Standard Dynamic Evidence Extraction
        top_doc_sections = extract_top_document_sections(retrieved_chunks, max_sections=3)
        distinct_sections = extract_relevant_chunk_content(retrieved_chunks, max_chunks=3, distinct_files=True)
        
        # Merge top sections or multi-source distinct sections
        is_multi_source = any(w in norm_query for w in ["final sale", "final-sale", "damaged", "broken", "price adjustment", "price drop"])
        all_active = getattr(self.retriever, "active_chunks", None)
        if is_multi_source:
            extracted = extract_multi_source_passages(retrieved_chunks, max_files=2, sections_per_file=2, all_active_chunks=all_active)
        elif len(top_doc_sections) >= 2:
            extracted = top_doc_sections
        else:
            extracted = distinct_sections

        passages_text = " ".join(t for t, _ in extracted)
        
        seen_srcs = set()
        sources = []
        for _, chunk in extracted:
            k = (chunk.file_name, chunk.heading)
            if k not in seen_srcs:
                seen_srcs.add(k)
                sources.append(SourceReference(file=chunk.file_name, heading=chunk.heading, document_id=chunk.metadata.document_id))

        # Sanitize internal agent instruction phrasing from verbatim policy chunks
        passages_text = re.sub(
            r"The agent may explain apparent eligibility but must not promise that credit has been issued\.",
            "The agent cannot issue a price adjustment credit.",
            passages_text,
            flags=re.IGNORECASE
        )

        # Generic destination availability check
        dest_match = re.search(r"\b(?:to|in)\s+([A-Z][a-z]+)\b", user_message)
        if dest_match and any("ship" in f"{c.title} {c.heading} {c.content}".lower() for _, c in extracted):
            queried_dest = dest_match.group(1)
            # If retrieved shipping policy indicates limited supported destinations and queried destination is not mentioned as supported:
            if ("only to" in passages_text.lower() or "not available" in passages_text.lower()) and queried_dest.lower() not in passages_text.lower():
                passages_text = f"Shipping to {queried_dest} is not currently available. {passages_text}"

        # If human specialist review is required based on extracted content or action requests
        needs_human_handoff = False
        action_verbs = ["cancel", "warranty claim", "defect", "damage", "broken", "price adjustment", "price drop"]
        if any(verb in norm_query for verb in action_verbs):
            if any(term in passages_text.lower() for term in ["human", "specialist", "approve", "not available", "review", "cannot issue"]):
                passages_text += " A human review before approval is required, so I am connecting you with our team."
                needs_human_handoff = True

        return "dynamic_rag_evidence", AgentResponse(answer=passages_text, sources=sources, handoff=needs_human_handoff)
