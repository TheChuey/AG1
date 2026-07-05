from datetime import datetime
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from chatbot.agents.search import SearchAgent
from chatbot.agents.scraper import ScraperAgent
from chatbot.agents.auditor import AuditorAgent
from chatbot.agents.summarizer import SummarizerAgent
from module import SessionDataManager


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
        self.session_mgr = SessionDataManager(storage_dir="sessions")
        self.graph = self._build_graph()

    def _search_node(self, state: PipelineState) -> dict:
        print(f"[ResearchPipeline] Node SEARCH | query={state['query'][:60]}")
        urls = self.search_agent.search(state["query"])
        print(f"[ResearchPipeline] Node SEARCH found {len(urls)} URLs")
        return {"discovered_urls": urls}

    def _scrape_node(self, state: PipelineState) -> dict:
        n = len(state["discovered_urls"])
        print(f"[ResearchPipeline] Node SCRAPE | {n} URLs")
        pages = self.scraper_agent.scrape_batch(state["discovered_urls"])
        ok = sum(1 for p in pages if p["content"])
        print(f"[ResearchPipeline] Node SCRAPE | {ok}/{n} succeeded")
        return {"scraped_pages": pages}

    def _audit_node(self, state: PipelineState) -> dict:
        n = len(state["scraped_pages"])
        print(f"[ResearchPipeline] Node AUDIT | {n} pages")
        approved = self.auditor_agent.audit_batch(state["scraped_pages"])
        print(f"[ResearchPipeline] Node AUDIT | {len(approved)}/{n} approved")
        return {"audited_pages": approved}

    def _summarize_node(self, state: PipelineState) -> dict:
        n = len(state["audited_pages"])
        print(f"[ResearchPipeline] Node SUMMARIZE | {n} pages")
        summaries = self.summarizer_agent.summarize_batch(state["audited_pages"])
        print(f"[ResearchPipeline] Node SUMMARIZE | {len(summaries)} summaries")
        return {"summaries": summaries}

    def _report_node(self, state: PipelineState) -> dict:
        print(f"[ResearchPipeline] Node REPORT | generating from {len(state['summaries'])} items")
        lines = []
        lines.append("# AI Research Report\n")
        lines.append(f"Generated: {datetime.now()}\n")
        lines.append(f"Query: {state['query']}\n\n---\n")
        scrape_prompts = []
        scrape_summaries = []
        for item in state["summaries"]:
            lines.append(f"## {item['title']}\n")
            lines.append(f"URL: {item['url']}\n\nSummary:\n{item['summary']}\n\n---\n")
            scrape_prompts.append(f"For URL {item['url']}, extract the main sections, data tables, and any FAQs that could help a user interested in the topic.")
            short = item['summary'][:150]
            scrape_summaries.append(f"{item['title']}: {short}...")
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
        print(f"[ResearchPipeline] run() invoked | query={query[:60]}")
        self.session_mgr.register_session(f"pipeline_{id(self)}", "research")
        result = self.graph.invoke({"query": query})
        print(f"[ResearchPipeline] run() complete")
        return result
