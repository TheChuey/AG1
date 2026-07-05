from datetime import datetime
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from chatbot.agents.search import SearchAgent
from chatbot.agents.scraper import ScraperAgent
from chatbot.agents.auditor import AuditorAgent
from chatbot.agents.summarizer import SummarizerAgent


class PipelineState(TypedDict):
    query: str
    discovered_urls: List[str]
    scraped_pages: List[dict]
    audited_pages: List[dict]
    summaries: List[dict]
    final_report: str


class ResearchPipeline:
    def __init__(self, model_name: str = "qwen2.5-coder", max_results: int = 5):
        self.search_agent = SearchAgent(max_results=max_results)
        self.scraper_agent = ScraperAgent()
        self.auditor_agent = AuditorAgent(model_name=model_name)
        self.summarizer_agent = SummarizerAgent(model_name=model_name)
        self.graph = self._build_graph()

    def _search_node(self, state: PipelineState) -> dict:
        urls = self.search_agent.search(state["query"])
        return {"discovered_urls": urls}

    def _scrape_node(self, state: PipelineState) -> dict:
        pages = self.scraper_agent.scrape_batch(state["discovered_urls"])
        return {"scraped_pages": pages}

    def _audit_node(self, state: PipelineState) -> dict:
        approved = self.auditor_agent.audit_batch(state["scraped_pages"])
        return {"audited_pages": approved}

    def _summarize_node(self, state: PipelineState) -> dict:
        summaries = self.summarizer_agent.summarize_batch(state["audited_pages"])
        return {"summaries": summaries}

    def _report_node(self, state: PipelineState) -> dict:
        lines = []
        lines.append("# AI Research Report\n")
        lines.append(f"Generated: {datetime.now()}\n")
        lines.append(f"Query: {state['query']}\n\n---\n")
        # Build prompt for deeper scraping and a concise summary per URL
        scrape_prompts = []
        scrape_summaries = []
        for item in state["summaries"]:
            # Truncate summary already done earlier
            lines.append(f"## {item['title']}\n")
            lines.append(f"URL: {item['url']}\n\nSummary:\n{item['summary']}\n\n---\n")
            # Prompt guiding scraper to extract more details
            scrape_prompts.append(f"For URL {item['url']}, extract the main sections, data tables, and any FAQs that could help a user interested in the topic.")
            # Short summary for the scraper to prioritize
            short = item['summary'][:150]
            scrape_summaries.append(f"{item['title']}: {short}...")
        # Combine prompts and summaries
        combined_prompt = "\n".join(scrape_prompts)
        combined_summary = "\n".join(scrape_summaries)
        return {
            "final_report": "\n".join(lines),
            "scrape_prompt": combined_prompt,
            "scrape_summary": combined_summary,
        }

    def _build_graph(self):
        graph = StateGraph(PipelineState)
        graph.add_node("search", self._search_node)
        graph.add_node("scrape", self._scrape_node)
        graph.add_node("audit", self._audit_node)
        graph.add_node("summarize", self._summarize_node)
        graph.add_node("report", self._report_node)
        graph.add_edge(START, "search")
        graph.add_edge("search", "scrape")
        graph.add_edge("scrape", "audit")
        graph.add_edge("audit", "summarize")
        graph.add_edge("summarize", "report")
        graph.add_edge("report", END)
        return graph.compile()

    def run(self, query: str) -> dict:
        return self.graph.invoke({"query": query})
