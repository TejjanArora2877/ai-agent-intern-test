"""Configuration management for the Aster & Row Support Agent."""

import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge-base"
DATA_DIR = PROJECT_ROOT / "data"
ORDERS_FILE = DATA_DIR / "orders.json"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"


def _clean_gemini_base_url(url: Optional[str]) -> str:
    """Ensure Gemini base URL is strictly Google's native REST endpoint root without /openai."""
    default = "https://generativelanguage.googleapis.com/v1beta"
    if not url or not url.strip():
        return default
    cleaned = url.strip().rstrip("/")
    if cleaned.endswith("/openai"):
        cleaned = cleaned[:-7].rstrip("/")
    return cleaned or default


class Settings(BaseModel):
    """Application settings loaded from environment or defaults."""
    llm_provider: str = Field(default_factory=lambda: os.getenv("LLM_PROVIDER", "gemini"))
    gemini_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))
    gemini_base_url: str = Field(
        default_factory=lambda: _clean_gemini_base_url(os.getenv("GEMINI_BASE_URL"))
    )
    gemini_model: str = Field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.6-flash"))
    debug_mode: bool = Field(default_factory=lambda: os.getenv("DEBUG_MODE", "false").lower() in ("true", "1", "yes"))
    max_retrieved_chunks: int = 6
    
    # Paths
    knowledge_base_dir: Path = KNOWLEDGE_BASE_DIR
    orders_file: Path = ORDERS_FILE
    evaluation_dir: Path = EVALUATION_DIR

    @property
    def is_live_llm_enabled(self) -> bool:
        """Returns True if Gemini live mode is configured with an API key."""
        return bool(self.llm_provider == "gemini" and self.gemini_api_key and self.gemini_api_key.strip())


settings = Settings()
