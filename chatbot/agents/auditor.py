from typing import List, Literal
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama


class AuditResult(BaseModel):
    url: str
    status: Literal["KEEP", "ELIMINATE"]
    confidence_score: float = Field(ge=0.0, le=100.0)
    reason: str


class AuditorAgent:
    def __init__(self, model_name: str = "qwen2.5-coder", temperature: float = 0.0):
        llm = ChatOllama(model=model_name, temperature=temperature)
        self.audit_llm = llm.with_structured_output(AuditResult)

    def audit(self, url: str, title: str = "", content: str = "") -> AuditResult:
        print(f"[AuditorAgent] >>> auditing {url[:80]}")
        prompt = f"""
You are a Content Audit Agent.

KEEP: documentation, tutorials, technical guides, research articles, educational content
ELIMINATE: login pages, cookie pages, privacy policies, terms of service, error pages, thin marketing pages

URL: {url}
TITLE: {title}
CONTENT: {content[:5000]}
"""
        result = self.audit_llm.invoke(prompt)
        print(f"[AuditorAgent] <<< {result.status} (confidence={result.confidence_score:.1f})")
        return result

    def audit_batch(self, pages: list) -> list:
        print(f"[AuditorAgent] auditing batch of {len(pages)} pages")
        approved = []
        for page in pages:
            result = self.audit(page["url"], page["title"], page["content"])
            if result.status == "KEEP":
                approved.append(page)
        print(f"[AuditorAgent] batch done ({len(approved)}/{len(pages)} approved)")
        return approved
