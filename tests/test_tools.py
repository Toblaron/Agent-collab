import asyncio
import json

import httpx
import pytest

from agent_collab.actions import MAX_SEARCHES_PER_MESSAGE, extract_actions
from agent_collab.agents import Agent, Message
from agent_collab.llm import Bid, OpenAICompatBackend
from agent_collab.room import SEARCH_AUTHOR, Room, RoomSettings
from agent_collab.search import (
    BraveSearcher,
    DuckDuckGoSearcher,
    MockSearcher,
    SearchError,
    SearxngSearcher,
    TavilySearcher,
    format_results,
    make_searcher,
)

BOTH = ("whiteboard", "search")


def run(coro):
    return asyncio.run(coro)


# ---- action extraction ----------------------------------------------------


def test_whiteboard_block_is_extracted_and_replaced():
    text = "Here's the plan.\n```whiteboard\n# Plan\n- step 1\n```\nThoughts?"
    clean, actions = extract_actions(text, BOTH)
    assert actions.whiteboard == "# Plan\n- step 1"
    assert clean == "Here's the plan.\n[updated the whiteboard]\nThoughts?"


def test_unterminated_whiteboard_block_still_counts():
    _, actions = extract_actions("ok\n```whiteboard\n# Draft\n- a", BOTH)
    assert actions.whiteboard == "# Draft\n- a"


def test_last_whiteboard_block_wins():
    _, actions = extract_actions("```whiteboard\nv1\n```\nactually\n```whiteboard\nv2\n```", BOTH)
    assert actions.whiteboard == "v2"


def test_searches_are_extracted_capped_and_deduped():
    text = "[[search: a]] [[ Search : b ]] [[search: a]] [[search: c]]"
    clean, actions = extract_actions(text, BOTH)
    assert actions.searches == ["a", "b"][:MAX_SEARCHES_PER_MESSAGE]
    assert "[searching: b]" in clean


def test_markers_are_ignored_without_the_tool():
    text = "```whiteboard\nx\n```\n[[search: y]]"
    clean, actions = extract_actions(text, ())
    assert actions.whiteboard is None and actions.searches == []
    assert clean == text


def test_system_prompt_only_describes_enabled_tools():
    assert "[[search:" in Agent("A", "a", "", tools=BOTH).system_prompt([])
    no_tools = Agent("A", "a", "", tools=()).system_prompt([])
    assert "[[search:" not in no_tools and "```whiteboard" not in no_tools


# ---- tools inside a room --------------------------------------------------


class OneShot:
    """First agent says `reply`, then everyone goes quiet."""

    def __init__(self, reply: str):
        self.reply = reply
        self.whiteboards_seen: list[str] = []

    async def bid(self, agent, roster, transcript, whiteboard=""):
        self.whiteboards_seen.append(whiteboard)
        return Bid(urgency=0.9 if transcript[-1].author == "Human" and agent.name == "A" else 0.0, reason="")

    async def speak(self, agent, roster, transcript, whiteboard=""):
        yield self.reply


class FakeSearcher:
    name = "fake"

    def __init__(self, fail=False):
        self.queries = []
        self.fail = fail

    async def search(self, query):
        self.queries.append(query)
        if self.fail:
            raise SearchError("fake is down")
        return (await MockSearcher().search(query))[:1]


def test_room_applies_whiteboard_and_shows_it_to_later_bids():
    async def go():
        backend = OneShot("Drafted.\n```whiteboard\n# Plan\n```")
        room = Room("r", [Agent("A", "a", ""), Agent("B", "b", "")], backend, RoomSettings(max_agent_turns=3))
        events = room.subscribe()
        await room.post_human("plan it")
        await room.wait_idle()
        return room, backend, [events.get_nowait() for _ in range(events.qsize())]

    room, backend, events = run(go())
    assert room.whiteboard == "# Plan" and room.whiteboard_by == "A"
    assert room.messages[-1].text == "Drafted.\n[updated the whiteboard]"
    assert any(e["type"] == "whiteboard" and e["text"] == "# Plan" for e in events)
    assert backend.whiteboards_seen[-1] == "# Plan"  # the follow-up bid round saw the new board


def test_room_runs_searches_and_posts_results():
    async def go(searcher):
        room = Room("r", [Agent("A", "a", "")], OneShot("Checking. [[search: rust vs go]]"),
                    RoomSettings(max_agent_turns=2), searcher=searcher)
        await room.post_human("which language?")
        await room.wait_idle()
        return room

    searcher = FakeSearcher()
    room = run(go(searcher))
    assert searcher.queries == ["rust vs go"]
    assert room.messages[1].text == "Checking. [searching: rust vs go]"
    assert room.messages[2].author == SEARCH_AUTHOR
    assert 'Results for "rust vs go"' in room.messages[2].text

    failed = run(go(FakeSearcher(fail=True)))
    assert "failed: fake is down" in failed.messages[2].text

    off = run(go(None))
    assert "Search is turned off" in off.messages[2].text


def test_agent_without_search_tool_does_not_search():
    searcher = FakeSearcher()

    async def go():
        room = Room("r", [Agent("A", "a", "", tools=())], OneShot("[[search: x]]"),
                    RoomSettings(max_agent_turns=1), searcher=searcher)
        await room.post_human("hi")
        await room.wait_idle()

    run(go())
    assert searcher.queries == []


def test_whiteboard_reaches_the_model_prompt(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"urgency": 0.1, "reason": ""}'}}]})

    agent = Agent("G", "g", "", provider="groq")
    backend = OpenAICompatBackend(httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    run(backend.bid(agent, [agent], [Message("Human", "hi")], whiteboard="# Shared plan"))
    assert "<whiteboard>\n# Shared plan\n</whiteboard>" in seen[0]["messages"][1]["content"]


# ---- search providers (fake HTTP) -----------------------------------------


def client_returning(**kwargs):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, **kwargs)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler)), seen


def test_tavily_parser():
    client, seen = client_returning(json={"results": [{"title": "T", "url": "https://t.dev", "content": "<b>snip</b>"}]})
    results = run(TavilySearcher("tvly-k", client).search("q"))
    assert [(r.title, r.url, r.snippet) for r in results] == [("T", "https://t.dev", "snip")]
    assert seen[0].headers["authorization"] == "Bearer tvly-k"


def test_brave_parser():
    client, seen = client_returning(json={"web": {"results": [{"title": "B", "url": "https://b.dev", "description": "d &amp; e"}]}})
    results = run(BraveSearcher("bk", client).search("q"))
    assert results[0].snippet == "d & e" and seen[0].headers["x-subscription-token"] == "bk"


def test_searxng_parser():
    client, seen = client_returning(json={"results": [{"title": "S", "url": "https://s.dev", "content": "c"}]})
    results = run(SearxngSearcher("http://searx.local/", client).search("q"))
    assert results[0].url == "https://s.dev"
    assert str(seen[0].url).startswith("http://searx.local/search?q=q&format=json")


def test_duckduckgo_parser_unwraps_redirect_links():
    page = (
        '<div class="result"><a rel="nofollow" class="result__a" '
        'href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fpage&amp;rut=x">Example <b>Page</b></a>'
        '<a class="result__snippet" href="#">An <b>example</b> snippet.</a></div>'
    )
    client, _ = client_returning(text=page)
    results = run(DuckDuckGoSearcher(client).search("q"))
    assert [(r.title, r.url, r.snippet) for r in results] == [("Example Page", "https://example.org/page", "An example snippet.")]


def test_search_http_errors_become_search_errors():
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(429)))
    with pytest.raises(SearchError, match="HTTP 429"):
        run(TavilySearcher("k", client).search("q"))


def test_make_searcher_selection(monkeypatch):
    monkeypatch.setenv("AGENT_COLLAB_SEARCH", "auto")
    for var in ("TAVILY_API_KEY", "BRAVE_API_KEY", "SEARXNG_URL"):
        monkeypatch.delenv(var, raising=False)
    assert make_searcher().name == "duckduckgo"
    monkeypatch.setenv("BRAVE_API_KEY", "b")
    assert make_searcher().name == "brave"
    monkeypatch.setenv("TAVILY_API_KEY", "t")
    assert make_searcher().name == "tavily"
    monkeypatch.setenv("AGENT_COLLAB_SEARCH", "off")
    assert make_searcher() is None
    monkeypatch.setenv("AGENT_COLLAB_SEARCH", "searxng")
    with pytest.raises(SystemExit):
        make_searcher()


def test_format_results():
    assert format_results("x", []) == 'No results for "x".'
