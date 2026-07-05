from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    model: str = "qwen2.5-coder"
    temperature: float = 0.3
    provider: str = "ollama"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    ollama_base_url: str = "http://localhost:11434"
    system_prompt_path: str = "prompts/system_prompt.md"
    chroma_persist_dir: str = "backend/resources/chroma_db"
    max_tokens: int = 4096
    top_p: float = 0.9
    think_level: int = 1  # 0 disabled, 1 or 2 for thinking levels
    trace_memory: list[dict] = field(default_factory=list)  # stores reasoning steps with source

    overrides: dict = field(default_factory=dict)

    def get(self, key: str, default=None):
        if key in self.overrides:
            return self.overrides[key]
        return getattr(self, key, default)
