# source.py
"""Deprecated — use ``module.py`` instead.

This file is a refactored variant of the canonical ``module.py`` classes with a
``_JSONStorageMixin``.  It is kept for reference but all new code should import
from ``module``.

  from module import SessionDataManager, PromptAndHistoryBridge, LLMManager, ChatManager
"""

import os
import json
import time
import http.client
from typing import Dict, List, Any, Optional


class _JSONStorageMixin:
    """Mixin providing lightweight JSON file I/O used by several managers.
    
    The methods are deliberately simple – they raise ``IOError`` on failure so
    callers can decide how to handle the situation.
    """

    @staticmethod
    def _write_json(path: str, payload: Any) -> None:
        """Write *payload* as pretty‑printed JSON to *path*.
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)

    @staticmethod
    def _read_json(path: str) -> Any:
        """Read JSON from *path* and return the parsed object.
        """
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


class SessionDataManager(_JSONStorageMixin):
    """Broker for session metadata across agents.

    Stores optional per‑session JSON files in ``self.session_dir`` and keeps an
    in‑memory map for fast look‑ups.
    """

    def __init__(self, storage_dir: str = "sessions") -> None:
        self.session_dir = storage_dir
        os.makedirs(self.session_dir, exist_ok=True)
        self._active_sessions: Dict[str, Dict[str, Any]] = {}

    def register_session(self, session_id: str, agent_type: str) -> None:
        self._active_sessions[session_id] = {"agent_type": agent_type, "status": "initialized"}

    def get_session_meta(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self._active_sessions.get(session_id)

    def purge_session(self, session_id: str) -> bool:
        if session_id in self._active_sessions:
            del self._active_sessions[session_id]
        file_path = os.path.join(self.session_dir, f"{session_id}.json")
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                return True
            except OSError:
                return False
        return True

    # Optional helper to persist a session dictionary – used by callers that need
    # a durable representation.
    def persist_session(self, session_id: str, data: Dict[str, Any]) -> None:
        self._write_json(os.path.join(self.session_dir, f"{session_id}.json"), data)


class PromptAndHistoryBridge(_JSONStorageMixin):
    """Handles CRUD for markdown prompts and JSON chat history.
    """

    def __init__(self, prompts_dir: str = "prompts", history_dir: str = "history") -> None:
        self.prompts_dir = prompts_dir
        self.history_dir = history_dir
        os.makedirs(self.prompts_dir, exist_ok=True)
        os.makedirs(self.history_dir, exist_ok=True)

    def save_file(self, file_id: str, content: str, folder_type: str = "history") -> bool:
        base = self.history_dir if folder_type == "history" else self.prompts_dir
        ext = "json" if folder_type == "history" else "md"
        path = os.path.join(base, f"{file_id}.{ext}")
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return True
        except IOError:
            return False

    def read_markdown_prompt(self, prompt_name: str) -> str:
        path = os.path.join(self.prompts_dir, f"{prompt_name}.md")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Prompt '{prompt_name}' not found.")
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def format_history_for_endpoint(self, session_id: str) -> List[Dict[str, str]]:
        path = os.path.join(self.history_dir, f"{session_id}.json")
        if not os.path.exists(path):
            return []
        try:
            return self._read_json(path)  # type: ignore[return-value]
        except (json.JSONDecodeError, IOError):
            return []


class LLMManager:
    """Thin wrapper around a local Ollama server.
    """

    def __init__(self, host: str = "localhost", port: int = 11434) -> None:
        self.host = host
        self.port = port

    def fetch_configured_models(self) -> List[str]:
        try:
            conn = http.client.HTTPConnection(self.host, self.port, timeout=5)
            conn.request("GET", "/api/tags")
            resp = conn.getresponse()
            if resp.status == 200:
                data = json.loads(resp.read().decode())
                return [m["name"] for m in data.get("models", [])]
            return []
        except Exception:
            return []

    def generate_response(self, payload: List[Dict[str, str]], model: str) -> Dict[str, Any]:
        available = self.fetch_configured_models()
        if model not in available and f"{model}:latest" not in available:
            return {"status": "error", "message": f"Model '{model}' unavailable. Download it first."}
        try:
            conn = http.client.HTTPConnection(self.host, self.port, timeout=30)
            headers = {"Content-Type": "application/json"}
            body = json.dumps({"model": model, "messages": payload, "stream": False})
            conn.request("POST", "/api/chat", body, headers)
            resp = conn.getresponse()
            if resp.status == 200:
                result = json.loads(resp.read().decode())
                return {"status": "success", "content": result.get("message", {}).get("content", "")}
            return {"status": "error", "message": f"Engine HTTP error: {resp.status}"}
        except Exception as e:
            return {"status": "error", "message": f"Connection failure: {e}"}


class ChatManager(_JSONStorageMixin):
    """Persistence layer for conversation logs with lightweight branching.
    """

    def __init__(self, storage_dir: str = "history") -> None:
        self.chat_dir = storage_dir
        os.makedirs(self.chat_dir, exist_ok=True)

    def save_session_state(self, session_id: str, messages: List[Dict[str, str]], active_branch: str = "main") -> bool:
        payload = {
            "session_id": session_id,
            "active_branch": active_branch,
            "updated_at": time.time(),
            "messages": messages,
        }
        try:
            self._write_json(os.path.join(self.chat_dir, f"{session_id}.json"), payload)
            return True
        except IOError:
            return False

    def get_recent_chats_log(self) -> List[Dict[str, Any]]:
        logs: List[Dict[str, Any]] = []
        for fname in os.listdir(self.chat_dir):
            if not fname.endswith('.json'):
                continue
            try:
                data = self._read_json(os.path.join(self.chat_dir, fname))
                messages = data.get('messages', [])
                preview = (messages[-1]["content"][:30] + "...") if messages else "Empty Chat"
                logs.append({
                    "session_id": data.get('session_id'),
                    "preview": preview,
                    "updated_at": data.get('updated_at', 0),
                })
            except Exception:
                continue
        return sorted(logs, key=lambda x: x['updated_at'], reverse=True)

    def create_chat_branch(self, parent_session_id: str, fork_at_message_index: int) -> str:
        parent_path = os.path.join(self.chat_dir, f"{parent_session_id}.json")
        if not os.path.exists(parent_path):
            raise FileNotFoundError("Parent chat does not exist.")
        parent_data = self._read_json(parent_path)
        sliced = parent_data.get('messages', [])[: fork_at_message_index + 1]
        new_id = f"branch_{int(time.time())}_{parent_session_id}"
        self.save_session_state(new_id, sliced, active_branch=new_id)
        return new_id

# ---------------------------------------------------------------------------
# Example usage (can be removed for production)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Quick sanity check – prints nothing on success, raises on error.
    sdm = SessionDataManager()
    sdm.register_session('demo', 'research')
    print('Session meta ->', sdm.get_session_meta('demo'))
    sdm.purge_session('demo')
    print('After purge ->', sdm.get_session_meta('demo'))

    phb = PromptAndHistoryBridge()
    phb.save_file('sample_prompt', '# Sample\nPrompt', folder_type='prompts')
    print('Prompt content ->', phb.read_markdown_prompt('sample_prompt'))

    cm = ChatManager()
    cm.save_session_state('chat_demo', [{"role": "user", "content": "Hi"}])
    print('Recent chats ->', cm.get_recent_chats_log())

    # LLMManager test – will only work if Ollama is running.
    # llm = LLMManager()
    # print('Models ->', llm.fetch_configured_models())
