"""Observability tracer for capturing intermediate decision states and outputs."""

from typing import List, Dict, Any, Optional
from src.models.schemas import DebugTrace, ChatMessage


class AgentTracer:
    """Helper for constructing and formatting structured observability traces."""

    @classmethod
    def create_trace(
        cls,
        user_message: str,
        conversation_history: List[ChatMessage],
        order_query_detected: bool = False,
        order_id_extracted: Optional[str] = None,
        order_tool_result: Optional[Dict[str, Any]] = None,
        retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
        conflict_detected: bool = False,
        model_mode: str = "mock",
        raw_model_response: Optional[str] = None,
        latency_ms: float = 0.0,
    ) -> DebugTrace:
        """Create a DebugTrace instance."""
        return DebugTrace(
            user_message=user_message,
            conversation_history=conversation_history,
            order_query_detected=order_query_detected,
            order_id_extracted=order_id_extracted,
            order_tool_result=order_tool_result,
            retrieved_chunks=retrieved_chunks or [],
            conflict_detected=conflict_detected,
            model_mode=model_mode,
            raw_model_response=raw_model_response,
            latency_ms=latency_ms,
        )
