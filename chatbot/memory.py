from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from typing import List


class Memory:
    def __init__(self):
        self.messages: List[BaseMessage] = []

    def add_human_message(self, text: str):
        self.messages.append(HumanMessage(content=text))

    def add_ai_message(self, text: str):
        self.messages.append(AIMessage(content=text))

    # Compatibility wrapper used by ChatBot
    def add_message(self, role: str, text: str):
        """Add a message to the conversation history.

        Args:
            role: Either "user" or "assistant".
            text: The message content.
        """
        if role == "user":
            self.add_human_message(text)
        elif role == "assistant":
            self.add_ai_message(text)
        else:
            raise ValueError(f"Unsupported role '{role}'. Use 'user' or 'assistant'.")

    def get_messages(self) -> List[BaseMessage]:
        return self.messages

    def clear(self):
        self.messages = []

    def __len__(self) -> int:
        return len(self.messages)
