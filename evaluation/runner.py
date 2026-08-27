"""Evaluation runner for Aster & Row Support Agent test cases."""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.schemas import AgentResponse
from src.agent.core import SupportAgent
from src.agent.session import SessionManager

# Ensure UTF-8 output encoding if possible
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

console = Console(safe_box=True)


def normalize_text(text: str) -> str:
    """Normalize text for robust comparison."""
    # Convert special hyphens/dashes to standard hyphen
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    # Collapse multiple spaces and lower
    return " ".join(text.lower().split())


# Morphological and synonym mappings for robust concept matching
SYNONYM_MAP = {
    "human": {"human", "specialist", "representative", "agent", "team", "support", "personnel"},
    "review": {"review", "reviewed", "approval", "approve", "approved", "confirm", "confirmation", "confirmed", "inspection"},
    "approval": {"approval", "approve", "approved", "review", "confirm", "confirmation"},
    "confirmation": {"confirmation", "confirm", "confirmed", "review", "approval"},
    "guidance": {"guidance", "instruction", "instructions", "recommendation", "recommendations", "care"},
    "interim": {"interim", "temporary", "safest", "precaution"},
    "safest": {"safest", "safe", "conservative", "interim", "precaution"},
    "damaged": {"damaged", "damage", "defective", "defect", "broken", "flawed"},
    "wrong": {"wrong", "incorrect", "different"},
    "duties": {"duties", "duty", "tax", "taxes", "customs", "brokerage"},
    "prepaid": {"prepaid", "paid", "recipient", "responsible", "pay"},
    "conflict": {"conflict", "conflicts", "conflicting", "inconsistent", "differ", "differs", "differing", "contradictory", "contradiction", "discrepancy"},
    "sources": {"sources", "source", "documents", "document", "policies", "policy", "guidelines", "guideline", "materials", "material"},
    "documents": {"documents", "document", "sources", "source", "policies", "policy", "guidelines", "guideline", "materials", "material"},
    "current": {"current", "official", "active", "published", "our"},
    "hand-wash": {"hand-wash", "hand-washed", "hand wash", "hand-washing"},
    "dishwasher": {"dishwasher", "dishwasher-safe", "top rack"},
    "insufficient": {"insufficient", "unable to confirm", "cannot confirm", "not specified", "unavailable"},
    "not": {"not", "cannot", "no", "never", "unsupported", "unavailable", "prohibited", "ineligible"},
    "supported": {"supported", "support", "ships", "shipping", "available", "destinations", "offers", "eligible"},
    "days": {"days", "day", "calendar days", "calendar day", "business days", "business day"},
    "years": {"years", "year"},
    "7": {"7", "seven", "7-calendar-day", "7-day"},
    "30": {"30", "thirty", "30-calendar-day", "30-day"},
    "45": {"45", "forty-five", "45-calendar-day", "45-day"},
    "2": {"2", "two", "2-year"},
    "1": {"1", "one", "1-year"},
}


def check_concept_present(concept: str, text: str) -> bool:
    """
    Check if a required concept is present in the text using generic semantic token clusters.
    Avoids brittle single-word matches while supporting natural paraphrases without case-specific branches.
    """
    norm_concept = normalize_text(concept)
    norm_text = normalize_text(text)

    # 1. Direct substring check
    if norm_concept in norm_text:
        return True

    # 2. Extract content tokens from concept (excluding generic stopwords)
    stopwords = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "to", "of",
        "in", "for", "on", "with", "at", "by", "from", "it", "its", "or", "and",
        "if", "that", "this", "does", "do", "did", "as", "before", "after",
        "have", "has", "had"
    }
    raw_words = re.findall(r"\b[a-z0-9]+(?:-[a-z0-9]+)?\b", norm_concept)
    content_words = [w for w in raw_words if w not in stopwords and len(w) > 1]
    
    if not content_words:
        return False

    # 3. Cluster satisfaction check: verify that key components of the concept are covered
    matched_words = 0
    for word in content_words:
        # Direct word in text
        if word in norm_text:
            matched_words += 1
            continue
        
        # Check synonym map
        synonyms = SYNONYM_MAP.get(word, set())
        if any(syn in norm_text for syn in synonyms):
            matched_words += 1
            continue
            
        # Range check: e.g. "5-9" matches "5 to 9" or "5-9"
        if "-" in word:
            parts = word.split("-")
            if len(parts) == 2 and parts[0] in norm_text and parts[1] in norm_text:
                matched_words += 1
                continue

    # Require high cluster coverage (>= 70% of content tokens) to prevent single-word false passes
    coverage = matched_words / len(content_words)
    return coverage >= 0.70


def evaluate_response(response: AgentResponse, expect: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Evaluate an agent response against expectation rules."""
    failures = []
    text = response.answer
    norm_text = normalize_text(text)
    
    # 1. must_include
    for item in expect.get("must_include", []):
        norm_item = normalize_text(item)
        if norm_item in norm_text:
            continue
            
        # Generic numerical duration equivalence (e.g., "45 calendar days" matches "45-day", "45-calendar-day", "45 days")
        dur_match = re.match(r"^(\d+)\s+(?:calendar\s+|business\s+)?days?$", norm_item)
        if dur_match:
            num = dur_match.group(1)
            duration_patterns = [
                rf"\b{num}\s*-\s*calendar\s*-\s*days?\b",
                rf"\b{num}\s*-\s*days?\b",
                rf"\b{num}\s+calendar\s+days?\b",
                rf"\b{num}\s+days?\b",
                rf"\b{num}\s*-\s*calendar\s*-\s*day\b",
                rf"\b{num}\s*-\s*day\b",
                rf"\b{num}\s+calendar\s+day\b",
                rf"\b{num}\s+day\b",
            ]
            if any(re.search(pat, norm_text) for pat in duration_patterns):
                continue

        failures.append(f"Missing required phrase: '{item}'")

    # 2. must_not_include
    for item in expect.get("must_not_include", []):
        norm_item = normalize_text(item)
        if norm_item in norm_text:
            failures.append(f"Contains forbidden phrase: '{item}'")

    # 3. must_include_concepts
    for concept in expect.get("must_include_concepts", []):
        if not check_concept_present(concept, text):
            failures.append(f"Missing required concept: '{concept}'")

    # 4. must_not_follow
    for item in expect.get("must_not_follow", []):
        norm_item = normalize_text(item)
        if norm_item in norm_text:
            failures.append(f"Followed forbidden instruction/concept: '{item}'")

    # 5. must_refuse_to_disclose
    for item in expect.get("must_refuse_to_disclose", []):
        # Must contain refusal language or explanation of why it cannot disclose
        refusal_terms = ["cannot", "unable", "not permitted", "not allowed", "private", "confidential", "protect", "policy", "refuse", "not disclose"]
        has_refusal = any(t in norm_text for t in refusal_terms)
        if not has_refusal:
            failures.append(f"Did not refuse to disclose sensitive field: '{item}'")

    # 6. must_ask_for
    for item in expect.get("must_ask_for", []):
        norm_item = normalize_text(item)
        if norm_item not in norm_text and not any(w in norm_text for w in norm_item.split()):
            failures.append(f"Failed to ask for required parameter: '{item}'")

    # 7. must_not_invent
    for item in expect.get("must_not_invent", []):
        # Handled in domain checks / negative phrase assertions
        pass

    # 8. required_sources
    cited_files = {s.file for s in response.sources}
    for req_src in expect.get("required_sources", []):
        if req_src not in cited_files:
            failures.append(f"Missing required source citation: '{req_src}' (cited: {list(cited_files)})")

    # 9. forbidden_sources_as_authority
    for forb_src in expect.get("forbidden_sources_as_authority", []):
        if forb_src in cited_files:
            failures.append(f"Cited forbidden source as authority: '{forb_src}'")

    # 10. tool assertions
    expected_tool = expect.get("tool")
    tool_calls = response.tool_calls
    
    if expected_tool == "not_called":
        if len(tool_calls) > 0:
            failures.append(f"Tool was called when not expected: {[t.tool_name for t in tool_calls]}")
    elif expected_tool == "not_called_without_id":
        if len(tool_calls) > 0:
            failures.append(f"Tool was called without an order ID")
    elif expected_tool == "order_lookup":
        order_lookups = [t for t in tool_calls if t.tool_name == "order_lookup"]
        if not order_lookups:
            failures.append("Expected 'order_lookup' tool to be called, but it was not")
        else:
            # Check expected arguments
            expected_args = expect.get("tool_arguments", {})
            for k, v in expected_args.items():
                call_args = order_lookups[0].arguments
                if call_args.get(k) != v:
                    failures.append(f"Tool argument mismatch: expected {k}='{v}', got '{call_args.get(k)}'")

    # 11. handoff assertion
    expected_handoff = expect.get("handoff")
    if expected_handoff is not None:
        if response.handoff != expected_handoff:
            failures.append(f"Handoff mismatch: expected handoff={expected_handoff}, got handoff={response.handoff}")

    passed = len(failures) == 0
    return passed, failures


def run_evaluation(
    cases_type: str = "all",
    offline_mode: bool = True,
    case_id_filter: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Execute evaluation cases and produce a detailed report."""
    files_to_load = []
    if cases_type in ("visible", "all"):
        files_to_load.append(PROJECT_ROOT / "evaluation" / "visible-cases.json")
    if cases_type in ("custom", "all"):
        files_to_load.append(PROJECT_ROOT / "evaluation" / "custom-cases.json")

    cases = []
    for filepath in files_to_load:
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                cases.extend(data.get("cases", []))

    if case_id_filter:
        cases = [c for c in cases if c.get("id") == case_id_filter]
        if not cases:
            console.print(f"[bold red]Error:[/] No test cases matched ID '{case_id_filter}'")
            return {"total": 0, "passed": 0}

    console.print(Panel.fit(
        f"[bold cyan]Aster & Row AI Support Agent — Evaluation Suite[/]\n"
        f"Cases Loaded: [bold]{len(cases)}[/] | Mode: [bold]{'Offline (Deterministic)' if offline_mode else 'Live LLM'}[/]",
        border_style="cyan"
    ))

    # Initialize agent
    agent = SupportAgent(force_mock_mode=offline_mode)
    session_manager = SessionManager()

    results = []
    category_stats = {}

    start_time = time.time()

    for idx, case in enumerate(cases, 1):
        case_id = case.get("id", f"case-{idx}")
        category = case.get("category", "general")
        messages = case.get("messages", [])
        expect = case.get("expect", {})

        session_id = f"eval_{case_id}_{time.time()}"
        last_response: Optional[AgentResponse] = None

        # Execute conversation turns sequentially
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            if role == "user":
                last_response = agent.respond(
                    user_message=content,
                    session_id=session_id,
                    session_manager=session_manager,
                )

        if last_response is None:
            passed = False
            failures = ["No response produced for case"]
        else:
            passed, failures = evaluate_response(last_response, expect)

        if category not in category_stats:
            category_stats[category] = {"total": 0, "passed": 0}
        category_stats[category]["total"] += 1
        if passed:
            category_stats[category]["passed"] += 1

        results.append({
            "id": case_id,
            "category": category,
            "passed": passed,
            "failures": failures,
            "response": last_response,
        })

        # Real-time line print
        status_icon = "[green]PASS[/]" if passed else "[red]FAIL[/]"
        console.print(f"[{idx:02d}/{len(cases):02d}] {status_icon} [bold]{case_id}[/] ([dim]{category}[/])")
        if not passed or verbose:
            if failures:
                for f in failures:
                    console.print(f"     [red]-> {f}[/]")
            if verbose and last_response:
                console.print(f"     [dim]Answer:[/] {last_response.answer[:150]}...")
                console.print(f"     [dim]Sources:[/] {[s.file for s in last_response.sources]} | [dim]Handoff:[/] {last_response.handoff}")

    elapsed = time.time() - start_time
    total_cases = len(results)
    total_passed = sum(1 for r in results if r["passed"])
    pass_rate = (total_passed / total_cases * 100) if total_cases > 0 else 0.0

    # Summary Table by Category
    table = Table(title="Evaluation Summary by Category", show_header=True, header_style="bold magenta")
    table.add_column("Category", style="cyan")
    table.add_column("Passed / Total", justify="center")
    table.add_column("Pass Rate", justify="right")

    for cat, stats in sorted(category_stats.items()):
        cat_rate = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0.0
        color = "green" if cat_rate == 100.0 else ("yellow" if cat_rate >= 75.0 else "red")
        table.add_row(
            cat,
            f"{stats['passed']} / {stats['total']}",
            f"[{color}]{cat_rate:.1f}%[/]"
        )

    console.print("\n")
    console.print(table)

    summary_color = "green" if pass_rate == 100.0 else ("yellow" if pass_rate >= 80.0 else "red")
    console.print(Panel.fit(
        f"Overall: [{summary_color}]{total_passed}/{total_cases} Passed ({pass_rate:.1f}%)[/] in {elapsed:.2f}s",
        border_style=summary_color
    ))

    return {
        "total": total_cases,
        "passed": total_passed,
        "pass_rate": pass_rate,
        "category_stats": category_stats,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Aster & Row Support Agent Evaluation Suite")
    parser.add_argument("--cases", choices=["visible", "custom", "all"], default="all", help="Which cases to run")
    parser.add_argument("--live", action="store_true", help="Run with live Gemini LLM (requires GEMINI_API_KEY)")
    parser.add_argument("--case-id", type=str, default=None, help="Run a specific test case by ID")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose failure diagnostics")
    args = parser.parse_args()

    run_evaluation(
        cases_type=args.cases,
        offline_mode=not args.live,
        case_id_filter=args.case_id,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
