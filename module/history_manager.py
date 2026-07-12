from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

class HistoryManager:
    def __init__(self):
        # { session_id: InMemoryChatMessageHistory }
        self.store = {}

    def get_session_history(self, session_id: str) -> InMemoryChatMessageHistory:
        """Get or create session history."""
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]

    def add_user_message(self, session_id: str, message: str):
        """Add user message."""
        history = self.get_session_history(session_id)
        history.add_user_message(message)

    def add_ai_message(self, session_id: str, message: str):
        """Add AI message."""
        history = self.get_session_history(session_id)
        history.add_ai_message(message)

    def get_recent_messages(self, session_id: str, limit: int = 4) -> list:
        """Get recent messages."""
        history = self.get_session_history(session_id)
        all_messages = history.messages
        return all_messages[-limit:] if all_messages else []

    def clear_history(self, session_id: str):
        """Clear history."""
        if session_id in self.store:
            self.store[session_id].clear()