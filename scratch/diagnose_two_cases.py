"""Direct live Gemini diagnostic runner for the two failing cases with rate limit backoff."""

import json
import os
import re
import sys
import time
import httpx
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load .env into os.environ if present
env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip().strip("\"'")

from src import config
config.settings = config.Settings()
from src.config import settings

from src.models.schemas import AgentResponse, SourceReference
from src.agent.core import SupportAgent
from src.agent.session import SessionManager
from src.agent.prompt import SYSTEM_PROMPT, build_agent_prompt
from src.agent.validator import DeterministicValidator
from src.rag.conflict import ConflictDetector
from src.tools.order_tool import extract_order_id
from evaluation.runner import evaluate_response


CASES_TO_DIAGNOSE = [
    {
        "id": "final-sale-damaged-exception",
        "category": "multi-source-grounding",
        "messages": [
            {
                "role": "user",
                "content": "A final-sale bag arrived with a broken zipper yesterday. Am I completely out of luck?"
            }
        ],
        "expect": {
            "must_include_concepts": [
                "final sale does not block damaged-item review",
                "report within 7 days",
                "human review before approval"
            ],
            "required_sources": [
                "03-final-sale-and-promotions.md",
                "04-damaged-or-wrong-items.md"
            ],
            "tool": "not_called",
            "handoff": True
        }
    },
    {
        "id": "genuine-active-source-conflict",
        "category": "source-conflict",
        "messages": [
            {
                "role": "user",
                "content": "Can I put the entire Breeze Tumbler in the dishwasher?"
            }
        ],
        "expect": {
            "must_include_concepts": [
                "current official sources conflict",
                "one says hand-wash the body",
                "one says all components are dishwasher safe",
                "human confirmation or safest interim guidance"
            ],
            "required_sources": [
                "11-product-care.md",
                "12-breeze-tumbler-product-card.md"
            ],
            "must_not_silently_choose_one": True,
            "tool": "not_called",
            "handoff": True
        }
    }
]


def call_gemini_raw_with_retry(prompt_text: str, max_retries: int = 5) -> str:
    """Call native Gemini REST API directly with exponential backoff on 429."""
    url = f"{settings.gemini_base_url}/models/{settings.gemini_model}:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": settings.gemini_api_key,
    }
    payload = {
        "system_instruction": {
            "parts": [{"text": SYSTEM_PROMPT}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt_text}]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json"
        }
    }

    for attempt in range(1, max_retries + 1):
        try:
            with httpx.Client(timeout=30.0) as client:
                res = client.post(url, headers=headers, json=payload)
                if res.status_code == 429:
                    wait_time = attempt * 8
                    print(f"Rate limited (429). Waiting {wait_time}s before retry {attempt}/{max_retries}...")
                    time.sleep(wait_time)
                    continue
                res.raise_for_status()
                data = res.json()
                candidates = data.get("candidates", [])
                if not candidates:
                    raise ValueError(f"No candidates returned: {data}")
                parts = candidates[0].get("content", {}).get("parts", [])
                if not parts:
                    raise ValueError(f"Candidate has no parts: {candidates[0]}")
                return parts[0].get("text", "")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429 and attempt < max_retries:
                wait_time = attempt * 8
                print(f"Rate limited (429). Waiting {wait_time}s before retry {attempt}/{max_retries}...")
                time.sleep(wait_time)
                continue
            raise
    raise RuntimeError("Failed to get response after retries")


def run_single_live_case(case):
    case_id = case["id"]
    print("\n" + "=" * 90)
    print(f"DIAGNOSING LIVE CASE: {case_id}")
    print("=" * 90)

    agent = SupportAgent(force_mock_mode=False)
    session_manager = SessionManager()
    session_id = f"diag_{case_id}"

    for msg in case["messages"]:
        content = msg["content"]
        history = session_manager.get_history(session_id)
        current_order_id = extract_order_id(content)

        order_view = None
        tool_called = False
        if current_order_id:
            order_view = agent.order_tool.lookup(current_order_id)
            tool_called = True

        retrieved_chunks = agent.retriever.retrieve(content, top_k=settings.max_retrieved_chunks)
        is_conflict, conflict_chunks, conflict_text = ConflictDetector.detect_conflict(retrieved_chunks, content)

        prompt_evidence = build_agent_prompt(
            user_query=content,
            conversation_history=history,
            retrieved_chunks=retrieved_chunks,
            order_view=order_view.model_dump() if order_view else None,
            order_missing=False,
        )

        # Call live Gemini directly with rate limit handling
        raw_json_str = call_gemini_raw_with_retry(prompt_evidence)
        
        # Parse into AgentResponse
        cleaned = raw_json_str.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        parsed_dict = json.loads(cleaned)

        sources = []
        for s in parsed_dict.get("sources", []):
            sources.append(
                SourceReference(
                    file=s.get("file", ""),
                    heading=s.get("heading", ""),
                    document_id=s.get("document_id"),
                )
            )

        pre_val_response = AgentResponse(
            answer=parsed_dict.get("answer", ""),
            sources=sources,
            handoff=bool(parsed_dict.get("handoff", False)),
        )

        # Deterministic Validator Step
        post_val_response = DeterministicValidator.validate_and_sanitize(
            response=pre_val_response,
            user_query=content,
            order_view=order_view,
            retrieved_chunks=retrieved_chunks,
            is_conflict=is_conflict,
        )

        # Evaluator Step
        passed, failures = evaluate_response(post_val_response, case["expect"])

        print(f"\n1. EXACT USER MESSAGE(S):\n   \"{content}\"")
        print(f"\n2. RETRIEVED CHUNKS ({len(retrieved_chunks)}):")
        for idx, c in enumerate(retrieved_chunks, 1):
            print(f"   [{idx}] {c.file_name} > {c.heading} (score: {c.score:.2f})")
        print(f"\n3. ORDER TOOL CALL: {tool_called} (order_id: {current_order_id})")
        print(f"\n4. CONFLICT DETECTOR RESULT:")
        print(f"   - is_conflict: {is_conflict}")
        if conflict_chunks:
            print(f"   - conflict_chunks: {[c.file_name + ' > ' + c.heading for c in conflict_chunks]}")
        else:
            print(f"   - conflict_chunks: []")
        print(f"\n5. EXACT EVIDENCE PLACED INTO GEMINI PROMPT:\n{prompt_evidence}")
        print(f"\n6. EXACT GEMINI JSON RESPONSE (RAW):\n{raw_json_str}")
        print(f"\n7. PARSED AGENT RESPONSE BEFORE VALIDATION:")
        print(f"   - answer:  {pre_val_response.answer}")
        print(f"   - sources: {[s.file + ' > ' + s.heading for s in pre_val_response.sources]}")
        print(f"   - handoff: {pre_val_response.handoff}")
        print(f"\n8. AGENT RESPONSE AFTER DETERMINISTIC VALIDATOR:")
        print(f"   - answer:  {post_val_response.answer}")
        print(f"   - sources: {[s.file + ' > ' + s.heading for s in post_val_response.sources]}")
        print(f"   - handoff: {post_val_response.handoff}")
        print(f"\n9. EXACT EVALUATOR ASSERTIONS THAT FAIL:")
        if passed:
            print("   (None - PASSED)")
        else:
            for f in failures:
                print(f"   - {f}")


if __name__ == "__main__":
    for idx, c in enumerate(CASES_TO_DIAGNOSE):
        if idx > 0:
            print("\nPausing 10s between calls for rate limiting...")
            time.sleep(10)
        run_single_live_case(c)
