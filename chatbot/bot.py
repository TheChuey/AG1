from typing import Optional
from chatbot.memory import Memory
from chatbot.prompt import PromptManager
from chatbot.history import HistoryManager
from chatbot.llm import LLMManager
from chatbot.config import Config


class ChatBot:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.memory = Memory()
        self.prompts = PromptManager()
        self.history = HistoryManager()
        self.llm = LLMManager(self.config)
        self.current_session_id = self.history.create_session()
        self.mode = "normal"  # possible: normal, business, thinking

    def set_mode(self, mode: str, level: int = None):
        """Set the agent mode and optional thinking level.
        Accepted mode values: normal, business, thinking.
        Level can be 0 (disabled), 1, or 2."""
        self.mode = mode if mode in ("normal", "business", "thinking") else "normal"
        if level is not None:
            self.config.think_level = level

    def chat(self, user_input: str, session_id: Optional[str] = None) -> str:
        """Process a user message and optionally switch to a specific session.
        Args:
            user_input: The message from the user.
            session_id: If provided, the bot will switch to this session before processing.
        """
        if session_id is not None:
            self.switch_session(session_id)
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
        self.history.save_message(self.current_session_id, "user", user_input)
        self.history.save_message(self.current_session_id, "assistant", response)
        
        return response

    def load_prompt(self, path: str):
        self.prompts.load_from_markdown(path)

    def load_prompt_text(self, text: str):
        self.prompts.load_from_text(text)

    def reset(self):
        self.memory.clear()
        self.current_session_id = self.history.create_session()

    def switch_session(self, session_id: str):
        self.current_session_id = session_id
        self.memory.messages = self.history.get_session(session_id)
