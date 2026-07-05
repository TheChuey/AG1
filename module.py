# module.py
"""Central module aggregating core management classes for the multi-agent application.

This file defines five loosely-coupled classes that handle session tracking,
prompt/history I/O, LLM communication, chat persistence, and model inventory.
The classes are intended to be imported by the various agent implementations
and FastAPI endpoints.

Design notes
------------
* **Separation of Concerns (SoC)** – each class has a single responsibility.
* **No tight framework coupling** – only the Python standard library is used.
* **Clear input/output contracts** – methods accept/return native ``dict``/``list``
  structures that can be JSON‑serialised directly.
* **Naming isolation** – attribute names are deliberately distinct across the
  classes to avoid accidental shadowing when instances are used together.

Dependency / usage map
----------------------
+-------------------------+---------------------------+-----------------------------+
| Class                   | Used by                   | Depends on                  |
+-------------------------+---------------------------+-----------------------------+
| SessionDataManager      | agents, ChatBot, server   | nothing (stdlib only)       |
| PromptAndHistoryBridge  | ChatBot, server           | nothing (stdlib only)       |
| LLMManager              | server (health/models)    | nothing (stdlib + HTTP)     |
| ChatManager             | ChatBot, server           | nothing (stdlib only)       |
| ModelManager            | server (model inventory)  | LLMManager (local instance) |
+-------------------------+---------------------------+-----------------------------+
No class imports another class from this module at import time.  ``ModelManager``
creates a local ``LLMManager`` instance at call time.  All classes are designed
to be instantiated independently by whatever orchestrator (ChatBot, FastAPI,
etc.) needs them.
"""

import os
import json
import uuid
import time
import subprocess
import http.client
from typing import Dict, List, Any, Optional


class SessionDataManager:
    """Central broker for session metadata across agents.

    *Creates a directory for JSON session files and tracks active sessions in
    memory.*

    Attributes
    ----------
    session_storage_dir: str
        Directory where optional session files are persisted.
    _active_sessions: Dict[str, Dict[str, Any]]
        In‑memory map of ``session_id`` → metadata (agent type, status, …).
    """

    def __init__(self, storage_dir: str = "sessions") -> None:
        # Distinct name to avoid clash with other classes that also store data.
        self.session_storage_dir = storage_dir
        os.makedirs(self.session_storage_dir, exist_ok=True)
        self._active_sessions: Dict[str, Dict[str, Any]] = {}

    def register_session(self, session_id: str, agent_type: str) -> None:
        """Register a new session and bind it to a specific agent type.
        """
        self._active_sessions[session_id] = {
            "agent_type": agent_type,
            "status": "initialized",
        }

    def get_session_meta(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Return stored metadata for *session_id* or ``None`` if unknown.
        """
        return self._active_sessions.get(session_id)

    def purge_session(self, session_id: str) -> bool:
        """Delete in‑memory metadata and any persisted JSON file for *session_id*.
        Returns ``True`` on success (or if the session was already absent).
        """
        if session_id in self._active_sessions:
            del self._active_sessions[session_id]

        target_path = os.path.join(self.session_storage_dir, f"{session_id}.json")
        if os.path.exists(target_path):
            try:
                os.remove(target_path)
                return True
            except OSError:
                return False
        return True


class PromptAndHistoryBridge:
    """Utility for CRUD operations on prompt markdown files and in-memory session tracking.

    Handles two concerns:
    * **Prompts** — read/save markdown prompt files on disk.
    * **Sessions** — in-memory conversation state (create, save messages, retrieve,
      delete, list).  This replaces the old ``HistoryManager`` from ``chatbot.history``.

    ``history_dir`` is accepted for backward compatibility but no longer used;
    all session state is held in memory.  Use :class:`ChatManager` for disk
    persistence of conversation logs.
    """

    def __init__(self, prompts_dir: str = "prompts", history_dir: str = "history") -> None:
        self.prompts_dir = prompts_dir
        self._sessions: Dict[str, List[Dict[str, str]]] = {}
        os.makedirs(self.prompts_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Prompt file I/O
    # ------------------------------------------------------------------

    def save_prompt(self, file_id: str, content: str) -> bool:
        """Write *content* to ``{file_id}.md`` inside the prompts directory."""
        path = os.path.join(self.prompts_dir, f"{file_id}.md")
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return True
        except IOError:
            return False

    def read_markdown_prompt(self, prompt_name: str) -> str:
        """Return the raw markdown for a stored prompt template.
        Raises ``FileNotFoundError`` if the file does not exist.
        """
        path = os.path.join(self.prompts_dir, f"{prompt_name}.md")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Prompt instruction blueprint '{prompt_name}' not found.")
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    # ------------------------------------------------------------------
    # In-memory session management (replaces chatbot.history.HistoryManager)
    # ------------------------------------------------------------------

    def create_session(self) -> str:
        """Create a new in-memory session and return its unique ID."""
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = []
        print(f"[PromptAndHistoryBridge] create_session | session={session_id}")
        return session_id

    def save_message(self, session_id: str, role: str, content: str) -> None:
        """Append a message dict to the in-memory session log."""
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append({"role": role, "content": content})

    def get_session(self, session_id: str) -> List[Dict[str, str]]:
        """Return the message list for *session_id* (empty list if unknown)."""
        return self._sessions.get(session_id, [])

    def delete_session(self, session_id: str) -> None:
        """Remove *session_id* from in-memory tracking."""
        self._sessions.pop(session_id, None)

    def list_sessions(self) -> List[str]:
        """Return all known session IDs."""
        return list(self._sessions.keys())


class LLMManager:
    """Thin wrapper around a local Ollama server.

    It validates model availability before sending a request and always returns a
    serialisable ``dict`` with a ``status`` key.
    """

    def __init__(self, host: str = "localhost", port: int = 11434) -> None:
        self.host = host
        self.port = port

    def fetch_configured_models(self) -> List[str]:
        """Ask the Ollama API for the list of cached model names.
        """
        try:
            conn = http.client.HTTPConnection(self.host, self.port, timeout=5)
            conn.request("GET", "/api/tags")
            response = conn.getresponse()
            if response.status == 200:
                data = json.loads(response.read().decode())
                return [model["name"] for model in data.get("models", [])]
            return []
        except Exception:
            return []

    def generate_response(self, payload: List[Dict[str, str]], model: str) -> Dict[str, Any]:
        """Send a chat payload to *model* and return the generated text.

        The function first checks that *model* exists locally.  If not, an error
        dictionary is returned.
        """
        available = self.fetch_configured_models()
        if model not in available and f"{model}:latest" not in available:
            return {
                "status": "error",
                "message": f"Model '{model}' is unavailable. Please download it via your local engine first.",
            }
        try:
            conn = http.client.HTTPConnection(self.host, self.port, timeout=30)
            headers = {"Content-Type": "application/json"}
            body = json.dumps({"model": model, "messages": payload, "stream": False})
            conn.request("POST", "/api/chat", body, headers)
            response = conn.getresponse()
            if response.status == 200:
                result = json.loads(response.read().decode())
                return {
                    "status": "success",
                    "content": result.get("message", {}).get("content", ""),
                }
            return {"status": "error", "message": f"Engine HTTP error status: {response.status}"}
        except Exception as e:
            return {"status": "error", "message": f"Failed connection to model server: {str(e)}"}


class ChatManager:
    """Persistence layer for conversation logs with lightweight branching support.
    """

    def __init__(self, storage_dir: str = "history") -> None:
        # Use a distinct attribute name to keep it separate from PromptAndHistoryBridge.
        self.chat_storage_dir = storage_dir
        os.makedirs(self.chat_storage_dir, exist_ok=True)

    def save_session_state(self, session_id: str, messages: List[Dict[str, str]], active_branch: str = "main") -> bool:
        """Serialise *messages* to ``{session_id}.json`` inside the chat storage.
        """
        payload = {
            "session_id": session_id,
            "active_branch": active_branch,
            "updated_at": time.time(),
            "messages": messages,
        }
        try:
            path = os.path.join(self.chat_storage_dir, f"{session_id}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=4)
            return True
        except IOError:
            return False

    def get_recent_chats_log(self) -> List[Dict[str, Any]]:
        """Return a sorted list of recent chats for UI sidebars.
        Each entry contains ``session_id``, a short preview, and ``updated_at``.
        """
        logs: List[Dict[str, Any]] = []
        for file in os.listdir(self.chat_storage_dir):
            if file.endswith(".json"):
                try:
                    with open(os.path.join(self.chat_storage_dir, file), "r", encoding="utf-8") as f:
                        data = json.load(f)
                    messages = data.get("messages", [])
                    preview = (messages[-1]["content"][:30] + "...") if messages else "Empty Chat"
                    logs.append({
                        "session_id": data.get("session_id"),
                        "preview": preview,
                        "updated_at": data.get("updated_at", 0),
                    })
                except Exception:
                    continue
        return sorted(logs, key=lambda x: x["updated_at"], reverse=True)

    def create_chat_branch(self, parent_session_id: str, fork_at_message_index: int) -> str:
        """Clone a chat up to *fork_at_message_index* and store it under a new ID.
        """
        parent_path = os.path.join(self.chat_storage_dir, f"{parent_session_id}.json")
        if not os.path.exists(parent_path):
            raise FileNotFoundError("Cannot branch from a non-existent parent chat timeline.")
        with open(parent_path, "r", encoding="utf-8") as f:
            parent_data = json.load(f)
        sliced_messages = parent_data.get("messages", [])[: fork_at_message_index + 1]
        new_branch_id = f"branch_{int(time.time())}_{parent_session_id}"
        self.save_session_state(session_id=new_branch_id, messages=sliced_messages, active_branch=new_branch_id)
        return new_branch_id

class ModelManager:
    """Detect installed Ollama models via API, filesystem scan, or CLI fallback.

    This is a utility class that wraps :class:`LLMManager` and adds filesystem
    and CLI fallback strategies so callers get a model list even when the Ollama
    HTTP endpoint is unreachable.
    """

    @staticmethod
    def get_installed_models() -> List[str]:
        """Return a list of installed model identifiers.

        Strategy (first success wins):
        1. Ask the Ollama HTTP API via ``LLMManager.fetch_configured_models``.
        2. Scan the ``~/.ollama/models`` directory for model folders.
        3. Run ``ollama list --format json`` via CLI.
        """
        # Strategy 1 – HTTP API (fastest)
        llm = LLMManager()
        models = llm.fetch_configured_models()
        if models:
            return models

        # Strategy 2 – filesystem scan
        models_dir = os.path.expanduser("~/.ollama/models")
        if os.path.isdir(models_dir):
            try:
                candidates = []
                for name in os.listdir(models_dir):
                    path = os.path.join(models_dir, name)
                    if os.path.isdir(path) and name not in ("blobs", "manifests"):
                        if any(fname.startswith("modelfile") or fname.endswith(".gguf") for fname in os.listdir(path)):
                            candidates.append(name)
                if candidates:
                    return candidates
            except Exception:
                pass

        # Strategy 3 – CLI fallback
        try:
            result = subprocess.run(
                ["ollama", "list", "--format", "json"],
                capture_output=True, text=True, check=True,
            )
            data = json.loads(result.stdout)
            return [item.get("model", "").split(":")[0] for item in data if isinstance(item, dict) and "model" in item]
        except Exception:
            return []

# ---------------------------------------------------------------------------
# Usage example (can be removed in production)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Simple sanity check when the module is executed directly.
    sdm = SessionDataManager()
    sdm.register_session("test123", "research")
    print("Session meta:", sdm.get_session_meta("test123"))
    sdm.purge_session("test123")
    print("After purge:", sdm.get_session_meta("test123"))
