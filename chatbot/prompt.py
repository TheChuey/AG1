from pathlib import Path
from typing import List
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


class PromptManager:
    def __init__(self):
        self.template = None
        self.system_content = ""

    def load_from_markdown(self, file_path: str):
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {file_path}")
        self.system_content = path.read_text(encoding="utf-8")
        self.template = ChatPromptTemplate.from_messages([
            ("system", self.system_content),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])

    def load_from_text(self, system_text: str):
        self.system_content = system_text
        self.template = ChatPromptTemplate.from_messages([
            ("system", self.system_content),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])

    # ----------------------------------------------------------------------
    # Public API for ChatBot – returns the raw system prompt string.
    # ----------------------------------------------------------------------
    def get_prompt(self, mode: str = "normal") -> str:
        """Return the loaded system prompt.

        The *mode* argument is currently unused but kept for future
        extensions where different modes might use distinct prompts.
        """
        return self.system_content

    def build_prompt(self, user_input: str, history: List[BaseMessage]):
        if self.template is None:
            raise RuntimeError("No prompt template loaded. Call load_from_markdown() first.")
        return self.template.format_messages(input=user_input, history=history)
