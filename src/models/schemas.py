"""Pydantic schemas for the Aster & Row AI Support Agent."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata parsed from YAML frontmatter of a knowledge-base document."""
    document_id: str
    title: str
    status: str  # "active", "superseded", "draft", etc.
    policy_authority: str  # "official", "none", etc.
    customer_answering: bool = True
    effective_date: Optional[str] = None
    last_reviewed: Optional[str] = None
    audience: Optional[str] = "customer"
    supersedes: Optional[str] = None
    superseded_by: Optional[str] = None


class RetrievedChunk(BaseModel):
    """A single retrieved section chunk from the knowledge base."""
    chunk_id: str
    file_name: str
    title: str
    heading: str
    heading_hierarchy: List[str] = Field(default_factory=list)
    content: str
    score: float = 0.0
    metadata: DocumentMetadata


class SourceReference(BaseModel):
    """A citation for an authoritative source."""
    file: str
    heading: str
    document_id: Optional[str] = None


class OrderItemView(BaseModel):
    """Customer-safe view of an item inside an order."""
    sku: Optional[str] = None
    name: str
    quantity: int
    final_sale: bool


class CustomerSafeOrderView(BaseModel):
    """Customer-safe representation of an order lookup."""
    order_id: str
    membership_tier: Optional[str] = None
    items: List[OrderItemView] = Field(default_factory=list)
    placed_at: Optional[str] = None
    status: str
    status_updated_at: Optional[str] = None
    shipped_at: Optional[str] = None
    delivered_at: Optional[str] = None
    carrier: Optional[str] = None
    tracking_number: Optional[str] = None
    estimated_delivery: Optional[str] = None
    estimated_delivery_available: bool = True
    customer_safe_message: Optional[str] = None
    requires_support_review: bool = False
    handoff_recommended: bool = False
    message: Optional[str] = None


class ChatMessage(BaseModel):
    """A single message in the conversation session."""
    role: str  # "user", "assistant", "system"
    content: str


class ToolCallRecord(BaseModel):
    """Record of a tool invocation for observability and assertions."""
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None


class DebugTrace(BaseModel):
    """Observability trace capturing intermediate steps and decisions."""
    user_message: str
    conversation_history: List[ChatMessage] = Field(default_factory=list)
    order_query_detected: bool = False
    order_id_extracted: Optional[str] = None
    order_tool_result: Optional[Dict[str, Any]] = None
    retrieved_chunks: List[Dict[str, Any]] = Field(default_factory=list)
    conflict_detected: bool = False
    model_mode: str = "mock"
    raw_model_response: Optional[str] = None
    latency_ms: float = 0.0


class AgentResponse(BaseModel):
    """Final user-facing response produced by the agent."""
    answer: str
    sources: List[SourceReference] = Field(default_factory=list)
    handoff: bool = False
    tool_calls: List[ToolCallRecord] = Field(default_factory=list)
    debug_trace: Optional[DebugTrace] = None
