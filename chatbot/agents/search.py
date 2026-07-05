from typing import List
from urllib.parse import urlparse
from ddgs import DDGS


class SearchAgent:
    def __init__(self, max_results: int = 10):
        self.max_results = max_results

    @staticmethod
    def is_valid_url(url: str) -> bool:
        if not url:
            return False
        parsed = urlparse(url)
        return parsed.scheme in ("http", "https")

    def search(self, query: str) -> List[str]:
        urls = []
        try:
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=self.max_results)
                for r in results:
                    url = r.get("href") or r.get("url")
                    if url and self.is_valid_url(url):
                        urls.append(url)
        except Exception as e:
            print(f"[SearchAgent] Error: {e}")
        seen = set()
        clean = []
        for u in urls:
            if u not in seen:
                seen.add(u)
                clean.append(u)
        return clean
