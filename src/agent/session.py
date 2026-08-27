"""In-memory multi-turn session manager."""

from typing import Dict, List, Optional
from src.models.schemas import ChatMessage


class SessionManager:
    """Manages conversation sessions isolated by session_id."""

    def __init__(self, max_history_turns: int = 10):
        self.max_history_turns = max_history_turns
        self._sessions: Dict[str, List[ChatMessage]] = {}

    def get_history(self, session_id: str) -> List[ChatMessage]:
        """Get message history for a given session ID."""
        return list(self._sessions.get(session_id, []))

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Add a message to the session history."""
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        
        self._sessions[session_id].append(ChatMessage(role=role, content=content))
        
        # Limit history length to prevent context explosion
        if len(self._sessions[session_id]) > self.max_history_turns * 2:
            self._sessions[session_id] = self._sessions[session_id][-(self.max_history_turns * 2):]

    def clear(self, session_id: str) -> None:
        """Clear conversation history for a session."""
        self._sessions.pop(session_id, None)
