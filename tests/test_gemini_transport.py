"""Unit tests for the native Gemini REST transport layer."""

import json
from unittest.mock import patch, MagicMock
import httpx
import pytest

from src.config import settings, _clean_gemini_base_url
from src.agent.core import SupportAgent
from src.models.schemas import RetrievedChunk, DocumentMetadata, CustomerSafeOrderView


def test_native_gemini_request_format_and_response_parsing():
    """
    Verify that _generate_live_llm constructs the exact native Gemini generateContent
    REST payload and parses the returned candidate JSON structure into an AgentResponse.
    """
    mock_gemini_response_json = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": json.dumps({
                                "answer": "Regular customers have 30 calendar days from delivery to return an unused item.",
                                "sources": [
                                    {
                                        "file": "01-returns-policy-current.md",
                                        "heading": "Standard return window",
                                        "document_id": "RET-2026-01"
                                    }
                                ],
                                "handoff": False
                            })
                        }
                    ],
                    "role": "model"
                },
                "finishReason": "STOP"
            }
        ]
    }

    mock_http_response = MagicMock(spec=httpx.Response)
    mock_http_response.status_code = 200
    mock_http_response.json.return_value = mock_gemini_response_json
    mock_http_response.raise_for_status.return_value = None

    captured_requests = []

    def mock_post(url, *args, **kwargs):
        captured_requests.append({"url": url, "args": args, "kwargs": kwargs})
        return mock_http_response

    # Temporarily configure a test key
    with patch.object(settings, "gemini_api_key", "test-gemini-key-12345"):
        with patch.object(httpx.Client, "post", side_effect=mock_post):
            agent = SupportAgent(force_mock_mode=False)
            
            chunk = RetrievedChunk(
                chunk_id="RET-01#0",
                file_name="01-returns-policy-current.md",
                title="Returns Policy",
                heading="Standard return window",
                heading_hierarchy=["Returns Policy", "Standard return window"],
                content="Customers on the standard plan may request a return within 30 calendar days of delivery.",
                score=5.0,
                metadata=DocumentMetadata(
                    document_id="RET-2026-01",
                    title="Returns Policy",
                    status="active",
                    policy_authority="official",
                )
            )

            raw_res, response = agent._generate_live_llm(
                user_message="How long do I have to return an item?",
                history=[],
                retrieved_chunks=[chunk],
                order_view=None,
                order_missing=False
            )

            # 1. Assert endpoint URL is strictly native Gemini generateContent with gemini-3.6-flash and contains NO /openai/
            assert len(captured_requests) == 1
            req = captured_requests[0]
            expected_endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent"
            assert req["url"] == expected_endpoint
            assert "openai" not in req["url"].lower()
            assert req["kwargs"]["params"]["key"] == "test-gemini-key-12345"
            assert req["kwargs"]["headers"]["x-goog-api-key"] == "test-gemini-key-12345"

            # 2. Assert payload schema matches native Gemini format without deprecated sampling params
            payload = req["kwargs"]["json"]
            assert "system_instruction" in payload
            assert payload["system_instruction"]["parts"][0]["text"].startswith("You are the official AI Customer Support Agent")
            assert "contents" in payload
            assert payload["contents"][0]["role"] == "user"
            assert "<knowledge_base_evidence>" in payload["contents"][0]["parts"][0]["text"]
            assert payload["generationConfig"] == {"responseMimeType": "application/json"}
            assert "temperature" not in payload["generationConfig"]
            assert "top_p" not in payload["generationConfig"]
            assert "top_k" not in payload["generationConfig"]

            # 3. Assert response was parsed into typed AgentResponse
            assert "30 calendar days" in response.answer
            assert len(response.sources) == 1
            assert response.sources[0].file == "01-returns-policy-current.md"
            assert response.handoff is False


def test_base_url_strips_openai_suffix():
    """
    Verify that even if GEMINI_BASE_URL environment variable has a legacy /openai path,
    it is automatically sanitized to Google's native REST base URL.
    """
    dirty_urls = [
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "https://generativelanguage.googleapis.com/v1beta/openai",
        "https://generativelanguage.googleapis.com/v1beta/",
        "https://generativelanguage.googleapis.com/v1beta",
    ]
    expected = "https://generativelanguage.googleapis.com/v1beta"
    for dirty in dirty_urls:
        cleaned = _clean_gemini_base_url(dirty)
        assert cleaned == expected, f"Failed for {dirty}: got {cleaned}"
        assert "openai" not in cleaned.lower()


def test_native_gemini_fallback_on_api_error():
    """
    Verify that if the native Gemini API returns an error, the agent gracefully
    falls back to the deterministic offline mock with a diagnostic prefix.
    """
    with patch.object(settings, "gemini_api_key", "test-gemini-key-12345"):
        with patch.object(httpx.Client, "post", side_effect=httpx.HTTPStatusError("404 Not Found", request=MagicMock(), response=MagicMock())):
            agent = SupportAgent(force_mock_mode=False)
            raw_res, response = agent._generate_live_llm(
                user_message="How long do I have to return an item?",
                history=[],
                retrieved_chunks=[],
                order_view=None,
                order_missing=False
            )

            assert response.answer.startswith("(Live Gemini LLM unavailable:")
            assert len(response.answer) > 30


def test_dotenv_automatic_configuration_loading(tmp_path, monkeypatch):
    """
    Regression test proving that configuration is automatically loaded from
    a .env file without requiring manual shell environment variables.
    """
    from src.config import reload_settings, Settings

    # Create temporary .env file
    temp_env = tmp_path / ".env"
    temp_env.write_text(
        "LLM_PROVIDER=gemini\n"
        "GEMINI_API_KEY=test-dotenv-key-abc-987\n"
        "GEMINI_MODEL=gemini-3.6-flash\n"
        "DEBUG_MODE=true\n",
        encoding="utf-8"
    )

    # Clear environment variables so they aren't pre-set
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("DEBUG_MODE", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)

    # Reload settings from temp .env
    loaded_settings = reload_settings(env_file=temp_env, override=True)

    assert loaded_settings.gemini_api_key == "test-dotenv-key-abc-987"
    assert loaded_settings.gemini_model == "gemini-3.6-flash"
    assert loaded_settings.debug_mode is True
    assert loaded_settings.is_live_llm_enabled is True


def test_env_var_priority_over_dotenv(tmp_path, monkeypatch):
    """
    Regression test proving that explicit environment variables maintain higher
    priority than .env file variables (override=False).
    """
    from src.config import reload_settings

    # Create temporary .env file with a default model
    temp_env = tmp_path / ".env"
    temp_env.write_text(
        "GEMINI_MODEL=from-dotenv-file\n",
        encoding="utf-8"
    )

    # Explicitly set higher-priority environment variable
    monkeypatch.setenv("GEMINI_MODEL", "from-explicit-env-var")

    # Load with override=False (the default behavior)
    loaded_settings = reload_settings(env_file=temp_env, override=False)

    assert loaded_settings.gemini_model == "from-explicit-env-var"
