import uuid
from typing import List, Dict
from langchain_core.messages import BaseMessage


class HistoryManager:
    def __init__(self):
        self.sessions: Dict[str, List[BaseMessage]] = {}

    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = []
        return session_id

    def save_session(self, session_id: str, messages: List[BaseMessage]):
        self.sessions[session_id] = messages

    def save_message(self, session_id: str, role: str, content: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        if role == "user":
            from langchain_core.messages import HumanMessage
            self.sessions[session_id].append(HumanMessage(content=content))
        elif role == "assistant":
            from langchain_core.messages import AIMessage
            self.sessions[session_id].append(AIMessage(content=content))

    def get_session(self, session_id: str) -> List[BaseMessage]:
        return self.sessions.get(session_id, [])

    def delete_session(self, session_id: str):
        self.sessions.pop(session_id, None)

    def list_sessions(self) -> List[str]:
        return list(self.sessions.keys())
