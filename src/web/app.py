"""FastAPI backend application for Aster & Row Support Agent Web GUI."""

import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.agent.core import SupportAgent
from src.agent.session import SessionManager
from src.config import settings
from src.models.schemas import SourceReference

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="Aster & Row Support Agent API",
    description="Customer Support AI Agent Web API with RAG, Tool Execution, and Debug Observability",
    version="1.0.0",
)

# CORS middleware for local development flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared Session Manager and Agent instances
session_manager = SessionManager()
offline_agent = SupportAgent(force_mock_mode=True)
live_agent = SupportAgent(force_mock_mode=False)


# --- Request and Response Schemas ---

class ChatRequest(BaseModel):
    message: str = Field(..., description="User message content", min_length=1)
    session_id: Optional[str] = Field(default=None, description="Optional conversation session ID")
    mode: Optional[str] = Field(default="offline", description="Execution mode: 'offline' or 'live'")


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: List[SourceReference]
    handoff: bool
    tool_calls: List[Dict[str, Any]] = []
    debug_trace: Optional[Dict[str, Any]] = None


class NewSessionResponse(BaseModel):
    session_id: str


class SessionMessage(BaseModel):
    role: str
    content: str


class SessionHistoryResponse(BaseModel):
    session_id: str
    messages: List[SessionMessage]


class HealthResponse(BaseModel):
    status: str
    default_mode: str
    live_llm_configured: bool
    model: str
    provider: str


# --- Defensive Sanitization Helper ---

def _sanitize_trace_for_api(trace_dict: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Ensure no raw PII or internal fields ever reach API responses, even inside debug traces.
    """
    if not trace_dict:
        return None

    sanitized = dict(trace_dict)
    
    # If order_tool_result is present, ensure forbidden internal fields are stripped
    if "order_tool_result" in sanitized and isinstance(sanitized["order_tool_result"], dict):
        order_res = dict(sanitized["order_tool_result"])
        forbidden_fields = {"customer_email", "email", "shipping_address", "address", "risk_score", "internal_note", "notes", "warehouse_note"}
        for f in forbidden_fields:
            order_res.pop(f, None)
        sanitized["order_tool_result"] = order_res

    return sanitized


# --- API Routes ---

@app.get("/api/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Return system health status and active provider configuration."""
    return HealthResponse(
        status="ok",
        default_mode="offline",
        live_llm_configured=settings.is_live_llm_enabled,
        model=settings.gemini_model,
        provider=settings.llm_provider,
    )


@app.post("/api/session/new", response_model=NewSessionResponse)
def create_session() -> NewSessionResponse:
    """Generate a brand new session ID."""
    new_id = f"web_{uuid.uuid4().hex[:12]}"
    return NewSessionResponse(session_id=new_id)


@app.get("/api/session/{session_id}", response_model=SessionHistoryResponse)
def get_session_history(session_id: str) -> SessionHistoryResponse:
    """Retrieve full conversation history for the given session ID."""
    history = session_manager.get_history(session_id)
    messages = [SessionMessage(role=m.role, content=m.content) for m in history]
    return SessionHistoryResponse(session_id=session_id, messages=messages)


@app.delete("/api/session/{session_id}")
def clear_session_history(session_id: str) -> Dict[str, str]:
    """Clear conversation history for the given session ID."""
    session_manager.clear(session_id)
    return {"status": "cleared", "session_id": session_id}


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """
    Process a user message through the SupportAgent and return the structured response.
    Never accesses raw orders.json directly; relies exclusively on SupportAgent and OrderLookupTool.
    """
    user_msg = request.message.strip()
    if not user_msg:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message content cannot be empty."
        )

    session_id = request.session_id.strip() if request.session_id and request.session_id.strip() else f"web_{uuid.uuid4().hex[:12]}"
    mode = (request.mode or "offline").lower().strip()

    # Select agent based on requested mode
    if mode == "live" and settings.is_live_llm_enabled:
        agent = live_agent
    else:
        agent = offline_agent

    try:
        agent_response = agent.respond(
            user_message=user_msg,
            session_id=session_id,
            session_manager=session_manager,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent failed to generate a response: {str(e)}"
        )

    # Format tool calls
    tool_calls_data = []
    if agent_response.tool_calls:
        for tc in agent_response.tool_calls:
            # Defensive check on tool arguments/results
            tool_calls_data.append({
                "tool_name": tc.tool_name,
                "arguments": tc.arguments,
                "result": tc.result,
            })

    # Prepare sanitized debug trace
    trace_data = None
    if agent_response.debug_trace:
        raw_trace_dict = agent_response.debug_trace.model_dump()
        trace_data = _sanitize_trace_for_api(raw_trace_dict)

    return ChatResponse(
        session_id=session_id,
        answer=agent_response.answer,
        sources=agent_response.sources,
        handoff=agent_response.handoff,
        tool_calls=tool_calls_data,
        debug_trace=trace_data,
    )


# --- Static Files & Single Page App Serving ---

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def serve_spa():
    """Serve the single page application HTML."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"message": "Frontend static index.html not found. Build static files first."}
        )
    return FileResponse(str(index_file))
