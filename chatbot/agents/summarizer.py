from typing import Dict
from langchain_ollama import ChatOllama


class SummarizerAgent:
    def __init__(self, model_name: str = "qwen2.5-coder", temperature: float = 0.0):
        self.llm = ChatOllama(model=model_name, temperature=temperature)

    def summarize(self, page: Dict[str, str]) -> Dict[str, str]:
        prompt = f"""
Summarize the following article focusing on facts and practical value. Maximum 300 words.

TITLE: {page.get("title", "")}
CONTENT: {page.get("content", "")[:6000]}
"""
        try:
            response = self.llm.invoke(prompt)
            return {
                "url": page["url"],
                "title": page["title"],
                "summary": response.content.strip(),
            }
        except Exception as e:
            print(f"[SummarizerAgent] Error: {page['url']} -> {e}")
            return {"url": page["url"], "title": page["title"], "summary": "ERROR_GENERATING_SUMMARY"}

    def summarize_batch(self, pages: list) -> list:
        return [self.summarize(p) for p in pages]
