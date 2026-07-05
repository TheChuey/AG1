from typing import Dict, Optional
import requests
from bs4 import BeautifulSoup


class ScraperAgent:
    HEADERS = {"User-Agent": "Mozilla/5.0"}
    # Increased timeout to 30 seconds and add simple retry logic
    TIMEOUT = 30

    def scrape(self, url: str) -> Dict[str, str]:
        print(f"[ScraperAgent] >>> scraping {url[:100]}")
        attempts = 2
        for attempt in range(attempts):
            try:
                r = requests.get(url, headers=self.HEADERS, timeout=self.TIMEOUT)
                soup = BeautifulSoup(r.text, "html.parser")
                for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                    tag.decompose()
                title = soup.title.text.strip() if soup.title else ""
                text = " ".join(soup.get_text(separator=" ", strip=True))
                result = {"url": url, "title": title, "content": text[:12000]}
                print(f"[ScraperAgent] <<< scraped OK ({len(result['content'])} chars)")
                return result
            except Exception as e:
                print(f"[ScraperAgent] Attempt {attempt + 1}/{attempts} failed for {url}: {e}")
                if attempt == attempts - 1:
                    print(f"[ScraperAgent] <<< FAILED for {url}")
                    return {"url": url, "title": "", "content": ""}
                import time
                time.sleep(2)
        return {"url": url, "title": "", "content": ""}

    def scrape_batch(self, urls: list) -> list:
        print(f"[ScraperAgent] batch scraping {len(urls)} URLs")
        results = [self.scrape(u) for u in urls]
        print(f"[ScraperAgent] batch done ({sum(1 for r in results if r['content'])}) successful")
        return results
