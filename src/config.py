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


class Settings(BaseModel):
    """Application settings loaded from environment or defaults."""
    openai_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    openai_base_url: str = Field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    model_name: str = Field(default_factory=lambda: os.getenv("MODEL_NAME", "gpt-4o-mini"))
    debug_mode: bool = Field(default_factory=lambda: os.getenv("DEBUG_MODE", "false").lower() in ("true", "1", "yes"))
    max_retrieved_chunks: int = 6
    
    # Paths
    knowledge_base_dir: Path = KNOWLEDGE_BASE_DIR
    orders_file: Path = ORDERS_FILE
    evaluation_dir: Path = EVALUATION_DIR

    @property
    def is_live_llm_enabled(self) -> bool:
        """Returns True if a live LLM API key is configured."""
        return bool(self.openai_api_key and self.openai_api_key.strip())


settings = Settings()
