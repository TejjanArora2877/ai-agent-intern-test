"""Diagnostic script to run the 5 failed live cases individually and capture complete traces."""

import json
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output encoding on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.config import settings
from src.agent.core import SupportAgent, _is_active_order_followup
from src.agent.session import SessionManager
from src.agent.prompt import SYSTEM_PROMPT, build_agent_prompt
from src.agent.validator import DeterministicValidator
from src.rag.conflict import ConflictDetector
from src.tools.order_tool import extract_order_id
from evaluation.runner import evaluate_response


TARGET_CASES = [
    "trailplus-return-window",
    "final-sale-damaged-exception",
    "canada-multiturn",
    "genuine-active-source-conflict",
    "multiturn-order-return-eligibility",
]


def load_all_cases():
    cases_files = [
        PROJECT_ROOT / "evaluation" / "visible-cases.json",
        PROJECT_ROOT / "evaluation" / "custom-cases.json",
    ]
    all_cases = {}
    for fpath in cases_files:
        if fpath.exists():
            with open(fpath, "r", encoding="utf-8") as f:
                d = json.load(f)
                for c in d.get("cases", []):
                    all_cases[c["id"]] = c
    return all_cases


def diagnose_case(case):
    case_id = case["id"]
    print("=" * 80)
    print(f" DIAGNOSING CASE: {case_id} ({case.get('category')})")
    print("=" * 80)

    agent = SupportAgent(force_mock_mode=False)
    session_manager = SessionManager()
    session_id = f"diag_{case_id}"

    messages = case["messages"]
    expect = case.get("expect", {})

    last_response = None
    for turn_idx, msg in enumerate(messages, 1):
        role = msg["role"]
        content = msg["content"]
        print(f"\n--- Turn {turn_idx} [{role}]: \"{content}\" ---")

        # Capture pre-execution context
        history = session_manager.get_history(session_id)
        current_order_id = extract_order_id(content)
        session_order_id = current_order_id
        if not session_order_id:
            for past_msg in reversed(history):
                past_id = extract_order_id(past_msg.content)
                if past_id:
                    session_order_id = past_id
                    break

        order_view = None
        order_missing = False
        tool_called = False

        if current_order_id:
            order_view = agent.order_tool.lookup(current_order_id)
            tool_called = True
        elif session_order_id:
            tentative_view = agent.order_tool.lookup(session_order_id)
            if _is_active_order_followup(content, tentative_view):
                order_view = tentative_view
                tool_called = True

        norm_query = content.lower()
        is_anaphoric = bool(norm_query and any(w in norm_query for w in ["it", "this", "that", "these", "those", "what about", "how about"]))
        enriched_query = content
        if history and is_anaphoric:
            user_past_msgs = [m.content for m in history if m.role == "user"]
            if user_past_msgs:
                enriched_query = f"{' '.join(user_past_msgs)} {content}"

        is_pure_tracking = bool(
            order_view and 
            any(w in norm_query for w in ["where is", "track", "status of", "when will", "has it arrived"]) and 
            not any(w in norm_query for w in ["return", "policy", "warranty", "cancel", "refund"])
        )

        if order_view and order_view.items and not is_pure_tracking:
            item_details = " ".join(f"{item.name} {'final sale' if item.final_sale else ''}" for item in order_view.items)
            enriched_query = f"{enriched_query} {item_details} {order_view.membership_tier or ''}"

        if is_pure_tracking:
            retrieved_chunks = []
        else:
            retrieved_chunks = agent.retriever.retrieve(enriched_query, top_k=settings.max_retrieved_chunks)

        is_conflict, conflict_chunks, conflict_text = ConflictDetector.detect_conflict(retrieved_chunks, content)

        built_prompt = build_agent_prompt(
            user_query=content,
            conversation_history=history,
            retrieved_chunks=retrieved_chunks,
            order_view=order_view.model_dump() if order_view else None,
            order_missing=order_missing,
        )

        # Call live agent
        last_response = agent.respond(
            user_message=content,
            session_id=session_id,
            session_manager=session_manager,
        )

        trace = last_response.debug_trace

        print(f"\nA. User Message: {content}")
        print(f"B. Retrieved Chunks ({len(retrieved_chunks)}):")
        for c in retrieved_chunks:
            print(f"   - {c.file_name} > {c.heading} (score: {c.score:.2f})")
        print(f"C. Order Lookup Called: {tool_called} (current: {current_order_id}, session: {session_order_id})")
        print(f"D. Sanitized Order View: {order_view.model_dump() if order_view else None}")
        print(f"E. ConflictDetector Result: is_conflict={is_conflict}")
        if is_conflict:
            print(f"   Conflict Chunks: {[c.file_name + ' > ' + c.heading for c in (conflict_chunks or [])]}")
        print(f"F. Exact Prompt/Evidence supplied to Gemini:\n--- SYSTEM PROMPT ---\n{SYSTEM_PROMPT}\n--- USER PROMPT ---\n{built_prompt}")
        print(f"G. Raw Gemini JSON Response:\n{trace.raw_model_response if trace else 'N/A'}")
        print(f"H. Parsed AgentResponse Before Post-Validation:")
        # Reproduce pre-validation parsed response from raw JSON
        if trace and trace.raw_model_response:
            try:
                cleaned = trace.raw_model_response.strip()
                if cleaned.startswith("```"):
                    import re
                    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
                    cleaned = re.sub(r"\s*```$", "", cleaned)
                parsed_dict = json.loads(cleaned)
                print(json.dumps(parsed_dict, indent=2))
            except Exception as ex:
                print(f"   (Failed to parse: {ex})")
        print(f"I. AgentResponse After Deterministic Validation:")
        print(f"   Answer:  {last_response.answer}")
        print(f"   Sources: {[s.file + ' > ' + s.heading for s in last_response.sources]}")
        print(f"   Handoff: {last_response.handoff}")

    # Evaluate last turn
    passed, failures = evaluate_response(last_response, expect)
    print("\n" + "-" * 80)
    print(f"J. EVALUATION RESULT: {'PASS' if passed else 'FAIL'}")
    if not passed:
        for f in failures:
            print(f"   REASON: {f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    cases_dict = load_all_cases()
    for cid in TARGET_CASES:
        if cid in cases_dict:
            diagnose_case(cases_dict[cid])
        else:
            print(f"Case {cid} not found!")
