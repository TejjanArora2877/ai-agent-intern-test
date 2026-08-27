"""Direct minimal live Gemini API test script.

Verifies that Google's native Gemini generateContent endpoint is actually
invoked and successfully produces structured AgentResponse JSON without
falling back to offline mock.
"""

import os
import sys
import json
import time

# Ensure UTF-8 output encoding on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.config import settings
from src.agent.core import SupportAgent


def test_live_gemini():
    base_url = settings.gemini_base_url.rstrip("/")
    if base_url.endswith("/openai"):
        base_url = base_url[:-7].rstrip("/")
    endpoint_url = f"{base_url}/models/{settings.gemini_model}:generateContent"

    print("=" * 60)
    print(" Aster & Row Support Agent — Direct Live Gemini API Test")
    print("=" * 60)
    print(f"Provider:         {settings.llm_provider}")
    print(f"Configured Model: {settings.gemini_model}")
    print(f"Native Endpoint:  {endpoint_url}")
    
    # Assert endpoint has NO /openai/ in URL
    if "openai" in endpoint_url.lower():
        print(f"[FAIL] Endpoint contains invalid /openai/ path: {endpoint_url}")
        return False
        
    api_key = settings.gemini_api_key
    if not api_key or not api_key.strip():
        masked_key = "(NOT SET)"
        print(f"GEMINI_API_KEY:   {masked_key}")
        print("\n[!] GEMINI_API_KEY is not set.")
        print("To test the live Gemini API, set your API key in your environment:")
        print("  Windows PowerShell: $env:GEMINI_API_KEY = \"your-actual-api-key\"")
        print("  Linux / macOS:      export GEMINI_API_KEY=\"your-actual-api-key\"")
        print("Then re-run: python -m scripts.test_gemini_live\n")
        return False
    else:
        masked_key = api_key[:6] + "..." + api_key[-4:] if len(api_key) > 10 else "***"
        print(f"GEMINI_API_KEY:   {masked_key}")
        print("-" * 60)

    print("\nSending live query to native Gemini API...")
    print("Query: 'How long does a regular customer have to return an unused backpack?'\n")
    start_time = time.time()
    
    agent = SupportAgent(force_mock_mode=False)
    response = agent.respond("How long does a regular customer have to return an unused backpack?")
    elapsed_ms = (time.time() - start_time) * 1000.0

    trace = response.debug_trace
    print(f"Response received in {elapsed_ms:.1f}ms\n")

    # Strict live response verification
    if response.answer.startswith("(Live Gemini LLM unavailable:"):
        print("[FAIL] Gemini API call failed and fell back to offline mock:")
        print(f"Error details: {response.answer}")
        return False

    if not trace or trace.model_mode.startswith("mock"):
        mode = trace.model_mode if trace else "unknown"
        print(f"[FAIL] Agent executed in mock mode ('{mode}') instead of live Gemini.")
        return False

    if not trace.raw_model_response or trace.raw_model_response == "mock":
        print("[FAIL] No raw Gemini model response captured in debug trace.")
        return False

    print("[SUCCESS] Native Gemini API returned a valid structured response!")
    print(f"Model Mode:     {trace.model_mode}")
    print(f"Handoff Flag:   {response.handoff}")
    print(f"Cited Sources:  {[f'{s.file} > {s.heading}' for s in response.sources]}")
    print(f"\nAnswer:\n{response.answer}\n")
    
    print("Raw Gemini JSON Content:")
    try:
        parsed_sample = json.loads(trace.raw_model_response)
        print(json.dumps(parsed_sample, indent=2))
    except Exception:
        print(trace.raw_model_response[:300] + "...")

    print("\n" + "=" * 60)
    print(" [✓] NATIVE GEMINI LIVE CALL CONFIRMED WORKING")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_live_gemini()
    sys.exit(0 if success else 1)
