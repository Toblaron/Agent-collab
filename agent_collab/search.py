"""Web search for agents.

Pick a provider with AGENT_COLLAB_SEARCH (default `auto`):
  tavily     TAVILY_API_KEY   (free tier: tavily.com)
  brave      BRAVE_API_KEY    (free tier: brave.com/search/api)
  searxng    SEARXNG_URL      (self-hosted, JSON format enabled)
  duckduckgo no key           (best-effort HTML scrape; may get rate-limited)
  off        disable search
`auto` takes the first configured of tavily → brave → searxng → duckduckgo.
"""

from __future__ import annotations

import html
import os
import re
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import parse_qs, unquote, urlparse

import httpx

MAX_RESULTS = 5
SNIPPET_CHARS = 300


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


class SearchError(RuntimeError):
    pass


class Searcher(Protocol):
    name: str

    async def search(self, query: str) -> list[SearchResult]: ...


def _clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text or "")
    return re.sub(r"\s+", " ", html.unescape(text)).strip()[:SNIPPET_CHARS]


class HttpSearcher:
    name = "base"

    def __init__(self, client: httpx.AsyncClient | None = None):
        self._client = client

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=15, follow_redirects=True, headers={"User-Agent": "agent-collab/0.1"}
            )
        return self._client

    async def search(self, query: str) -> list[SearchResult]:
        try:
            return (await self._search(query))[:MAX_RESULTS]
        except httpx.HTTPStatusError as e:
            raise SearchError(f"{self.name} returned HTTP {e.response.status_code}") from e
        except httpx.HTTPError as e:
            raise SearchError(f"{self.name} unreachable ({type(e).__name__})") from e
        except (KeyError, ValueError, TypeError) as e:
            raise SearchError(f"{self.name} returned an unexpected response") from e

    async def _search(self, query: str) -> list[SearchResult]:
        raise NotImplementedError


class TavilySearcher(HttpSearcher):
    name = "tavily"

    def __init__(self, api_key: str, client: httpx.AsyncClient | None = None):
        super().__init__(client)
        self.api_key = api_key

    async def _search(self, query: str) -> list[SearchResult]:
        r = await self.client.post(
            "https://api.tavily.com/search",
            json={"query": query, "max_results": MAX_RESULTS},
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        r.raise_for_status()
        return [SearchResult(_clean(x.get("title", "")), x["url"], _clean(x.get("content", ""))) for x in r.json()["results"]]


class BraveSearcher(HttpSearcher):
    name = "brave"

    def __init__(self, api_key: str, client: httpx.AsyncClient | None = None):
        super().__init__(client)
        self.api_key = api_key

    async def _search(self, query: str) -> list[SearchResult]:
        r = await self.client.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": MAX_RESULTS},
            headers={"X-Subscription-Token": self.api_key, "Accept": "application/json"},
        )
        r.raise_for_status()
        results = r.json().get("web", {}).get("results", [])
        return [SearchResult(_clean(x.get("title", "")), x["url"], _clean(x.get("description", ""))) for x in results]


class SearxngSearcher(HttpSearcher):
    name = "searxng"

    def __init__(self, base_url: str, client: httpx.AsyncClient | None = None):
        super().__init__(client)
        self.base_url = base_url.rstrip("/")

    async def _search(self, query: str) -> list[SearchResult]:
        r = await self.client.get(f"{self.base_url}/search", params={"q": query, "format": "json"})
        r.raise_for_status()
        return [SearchResult(_clean(x.get("title", "")), x["url"], _clean(x.get("content", ""))) for x in r.json()["results"]]


class DuckDuckGoSearcher(HttpSearcher):
    """Scrapes the no-JS HTML endpoint. No key, but layout changes or rate limits can break it."""

    name = "duckduckgo"
    RESULT_RE = re.compile(
        r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>.*?'
        r'class="result__snippet"[^>]*>(.*?)</(?:a|td|div)>',
        re.DOTALL,
    )

    async def _search(self, query: str) -> list[SearchResult]:
        r = await self.client.post("https://html.duckduckgo.com/html/", data={"q": query})
        r.raise_for_status()
        results = []
        for href, title, snippet in self.RESULT_RE.findall(r.text):
            url = html.unescape(href)
            if "duckduckgo.com/l/" in url:  # redirect wrapper: real URL is in ?uddg=
                url = unquote(parse_qs(urlparse(url).query).get("uddg", [url])[0])
            if url.startswith("//"):
                url = "https:" + url
            results.append(SearchResult(_clean(title), url, _clean(snippet)))
        return results


class MockSearcher:
    name = "mock"

    async def search(self, query: str) -> list[SearchResult]:
        return [
            SearchResult(f"{query.title()}: an overview", "https://example.com/overview", f"Background on {query}."),
            SearchResult(f"Common pitfalls with {query}", "https://example.com/pitfalls", "Three mistakes teams make."),
        ]


def make_searcher() -> Searcher | None:
    choice = os.environ.get("AGENT_COLLAB_SEARCH", "auto").lower()
    if os.environ.get("AGENT_COLLAB_MOCK") == "1" and choice in ("auto", "mock"):
        return MockSearcher()
    builders = {
        "tavily": lambda: TavilySearcher(os.environ["TAVILY_API_KEY"]) if os.environ.get("TAVILY_API_KEY") else None,
        "brave": lambda: BraveSearcher(os.environ["BRAVE_API_KEY"]) if os.environ.get("BRAVE_API_KEY") else None,
        "searxng": lambda: SearxngSearcher(os.environ["SEARXNG_URL"]) if os.environ.get("SEARXNG_URL") else None,
        "duckduckgo": lambda: DuckDuckGoSearcher(),
    }
    if choice == "off":
        return None
    if choice == "auto":
        for build in builders.values():
            if searcher := build():
                return searcher
        return None
    if choice not in builders:
        raise SystemExit(f"AGENT_COLLAB_SEARCH={choice!r}: expected auto, off, or one of {sorted(builders)}")
    searcher = builders[choice]()
    if searcher is None:
        raise SystemExit(f"AGENT_COLLAB_SEARCH={choice} but its key/URL env var is not set")
    return searcher


def format_results(query: str, results: list[SearchResult]) -> str:
    if not results:
        return f'No results for "{query}".'
    lines = [f'Results for "{query}":']
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r.title} — {r.url}")
        if r.snippet:
            lines.append(f"   {r.snippet}")
    return "\n".join(lines)
