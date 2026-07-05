from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama
from chatbot.config import Config


class LLMManager:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.model: Optional[BaseChatModel] = None
        self._build_model()

    def _build_model(self):
        provider = self.config.provider
        if provider == "ollama":
            self.model = ChatOllama(
                model=self.config.model,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                num_predict=self.config.max_tokens,
                base_url=self.config.ollama_base_url,
            )
        elif provider == "openai":
            from langchain_openai import ChatOpenAI
            self.model = ChatOpenAI(
                model=self.config.model,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_tokens=self.config.max_tokens,
                api_key=self.config.api_key,
                base_url=self.config.api_base,
            )
        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            self.model = ChatAnthropic(
                model=self.config.model,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_tokens=self.config.max_tokens,
                api_key=self.config.api_key,
                base_url=self.config.api_base,
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def generate(self, system_prompt: str, messages: list, mode: str = "normal", think_level: int = 1) -> str:
        """Generate a response using the underlying LLM.

        Args:
            system_prompt: The system instruction (often a prompt template).
            messages: List of message dicts with ``role`` and ``content`` keys.
            mode: Currently unused – kept for API compatibility.
            think_level: Currently unused – kept for future extensions.
        """
        # Build a simple concatenated prompt. More sophisticated
        # formatting (e.g., using ChatPromptTemplate) could be added later.
        # The format expected by Ollama/OpenAI chat models is a list of
        # messages, so we forward the list directly to the underlying model.
        if self.model is None:
            raise RuntimeError("LLM model not initialized")
        # Most chat models accept a list of messages; we simply pass it.
        try:
            response = self.model.invoke(messages)
            # ``invoke`` may return a LangChain response object; extract text.
            if hasattr(response, "content"):
                return response.content
            return str(response)
        except Exception as e:
            # Log and re‑raise for upstream handling.
            import traceback, sys
            tb = traceback.format_exc()
            print(f"[LLM Generate Error] {e}\n{tb}", file=sys.stderr)
            raise


    def switch_model(self, model_name: str, provider: Optional[str] = None):
        self.config.model = model_name
        if provider:
            self.config.provider = provider
        self._build_model()
