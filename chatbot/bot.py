from typing import Optional
from chatbot.memory import Memory
from chatbot.prompt import PromptManager
from chatbot.llm import LLMManager
from chatbot.config import Config
from module import SessionDataManager, PromptAndHistoryBridge, ChatManager


class ChatBot:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.memory = Memory()
        self.prompts = PromptManager()
        self.llm = LLMManager(self.config)
        self.sessions = SessionDataManager(storage_dir="sessions")
        self.prompt_bridge = PromptAndHistoryBridge(prompts_dir="prompts", history_dir="history")
        self.chat_storage = ChatManager(storage_dir="history")
        self.current_session_id = self.prompt_bridge.create_session()
        self.sessions.register_session(self.current_session_id, "chat")
        self.mode = "normal"
        print(f"[ChatBot] initialized | session={self.current_session_id} mode={self.mode}")

    def set_mode(self, mode: str, level: int = None):
        self.mode = mode if mode in ("normal", "business", "thinking") else "normal"
        if level is not None:
            self.config.think_level = level
        print(f"[ChatBot] set_mode | mode={self.mode} level={level}")

    def chat(self, user_input: str, session_id: Optional[str] = None) -> str:
        if session_id is not None:
            self.switch_session(session_id)
        print(f"[ChatBot] chat() | session={self.current_session_id} mode={self.mode}")
        messages = self.memory.get_messages() + [{"role": "user", "content": user_input}]
        system_prompt = self.prompts.get_prompt(self.mode)

        response = self.llm.generate(
            system_prompt=system_prompt,
            messages=messages,
            mode=self.mode,
            think_level=self.config.think_level
        )

        self.memory.add_message("user", user_input)
        self.memory.add_message("assistant", response)
        self.prompt_bridge.save_message(self.current_session_id, "user", user_input)
        self.prompt_bridge.save_message(self.current_session_id, "assistant", response)

        serialized = []
        for m in self.memory.get_messages():
            role = "user" if getattr(m, "type", "") == "human" else "assistant"
            serialized.append({"role": role, "content": m.content})
        self.chat_storage.save_session_state(self.current_session_id, serialized)

        print(f"[ChatBot] response generated | session={self.current_session_id}")
        return response

    def load_prompt(self, path: str):
        self.prompts.load_from_markdown(path)
        print(f"[ChatBot] prompt loaded | path={path}")

    def load_prompt_text(self, text: str):
        self.prompts.load_from_text(text)
        print(f"[ChatBot] prompt loaded from text ({len(text)} chars)")

    def reset(self):
        self.memory.clear()
        self.current_session_id = self.prompt_bridge.create_session()
        self.sessions.register_session(self.current_session_id, "chat")
        print(f"[ChatBot] reset | new session={self.current_session_id}")

    def switch_session(self, session_id: str):
        self.current_session_id = session_id
        raw = self.prompt_bridge.get_session(session_id)
        self.memory.messages = []
        for msg in raw:
            if msg["role"] == "user":
                from langchain_core.messages import HumanMessage
                self.memory.messages.append(HumanMessage(content=msg["content"]))
            else:
                from langchain_core.messages import AIMessage
                self.memory.messages.append(AIMessage(content=msg["content"]))
        print(f"[ChatBot] switch_session | session={session_id}")
