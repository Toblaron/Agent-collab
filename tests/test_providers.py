import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from agent_collab.agents import Agent, AgentError, Message, agent_from_dict
from agent_collab.llm import Bid, MockBackend, OpenAICompatBackend, RoutingBackend, ThinkFilter, parse_bid
from agent_collab.room import Room, RoomSettings
from agent_collab.server import create_app


def run(coro):
    return asyncio.run(coro)


# ---- bid parsing ----------------------------------------------------------


@pytest.mark.parametrize(
    "text, urgency",
    [
        ('{"urgency": 0.7, "reason": "new idea"}', 0.7),
        ('```json\n{"urgency": 0.4, "reason": "x"}\n```', 0.4),
        ('<think>hmm {"urgency": 0.9}</think>{"urgency": 0.2, "reason": "meh"}', 0.2),
        ('Sure! {"urgency": 3, "reason": "over"}', 1.0),  # clamped
        ("urgency: 0.55 because I have a point", 0.55),
        ("I have nothing to say.", 0.0),
    ],
)
def test_parse_bid_is_lenient(text, urgency):
    assert parse_bid(text).urgency == pytest.approx(urgency)


def test_think_filter_handles_tags_split_across_chunks():
    f = ThinkFilter()
    chunks = ["Hel", "lo <thi", "nk>secret ", "reasoning</th", "ink> world", "<"]
    out = "".join(f.feed(c) for c in chunks) + f.flush()
    assert out == "Hello  world<"


# ---- OpenAI-compatible backend against a fake server ----------------------


def fake_provider(bid_text: str, stream_tokens: list[str], seen: list[dict]):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append({"url": str(request.url), "auth": request.headers.get("authorization"), "body": body})
        if body.get("stream"):
            sse = "".join(
                f"data: {json.dumps({'choices': [{'delta': {'content': t}}]})}\n\n" for t in stream_tokens
            ) + "data: [DONE]\n\n"
            return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={"choices": [{"message": {"content": bid_text}}]})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


GROQ_AGENT = Agent("Gro", "builder", "", provider="groq", model="llama-3.3-70b-versatile")


def test_compat_backend_bids_and_streams(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    seen: list[dict] = []
    backend = OpenAICompatBackend(fake_provider('{"urgency": 0.8, "reason": "I can build it"}',
                                                ["<think>plan</think>", "Let me ", "sketch it."], seen))

    async def go():
        bid = await backend.bid(GROQ_AGENT, [GROQ_AGENT], [Message("Human", "hi")])
        text = "".join([t async for t in backend.speak(GROQ_AGENT, [GROQ_AGENT], [Message("Human", "hi")])])
        return bid, text

    bid, text = run(go())
    assert bid == Bid(urgency=0.8, reason="I can build it")
    assert text == "Let me sketch it."
    assert seen[0]["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert seen[0]["auth"] == "Bearer test-key"
    assert seen[0]["body"]["model"] == "llama-3.3-70b-versatile"
    assert seen[0]["body"]["messages"][0]["content"].startswith("You are Gro, the builder")
    assert seen[1]["body"]["stream"] is True


def test_compat_backend_bid_errors_become_silence(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(429, json={})))
    bid = run(OpenAICompatBackend(client).bid(GROQ_AGENT, [GROQ_AGENT], []))
    assert bid.urgency == 0 and "rate limited" in bid.reason


def test_ollama_needs_no_key_and_honours_base_url(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://gpu-box:11434/v1")
    seen: list[dict] = []
    agent = Agent("Olly", "critic", "", provider="ollama", model="qwen3")
    run(OpenAICompatBackend(fake_provider('{"urgency": 0.1, "reason": "x"}', [], seen)).bid(agent, [agent], []))
    assert seen[0]["url"] == "http://gpu-box:11434/v1/chat/completions"
    assert seen[0]["auth"] is None


# ---- routing & resilience -------------------------------------------------


class Tagged(MockBackend):
    def __init__(self, tag):
        super().__init__(delay=0)
        self.tag = tag

    async def bid(self, agent, roster, transcript, whiteboard=""):
        return Bid(urgency=0.5, reason=self.tag)


def test_routing_sends_each_agent_to_its_provider_backend():
    router = RoutingBackend(claude=Tagged("claude"), compat=Tagged("compat"))
    claude_agent = Agent("C", "c", "")
    assert run(router.bid(claude_agent, [], [])).reason == "claude"
    assert run(router.bid(GROQ_AGENT, [], [])).reason == "compat"


class BrokenSpeaker:
    async def bid(self, agent, roster, transcript, whiteboard=""):
        return Bid(urgency=0.9 if agent.name == "Bad" else 0.6, reason="")

    async def speak(self, agent, roster, transcript, whiteboard=""):
        if agent.name == "Bad":
            raise RuntimeError("model not found")
        yield "fine"


def test_a_failing_agent_is_muted_and_others_carry_on():
    async def go():
        room = Room("r", [Agent("Bad", "x", ""), Agent("Good", "y", "")], BrokenSpeaker(), RoomSettings(max_agent_turns=3))
        q = room.subscribe()
        await room.post_human("go")
        await room.wait_idle()
        events = []
        while not q.empty():
            events.append(q.get_nowait())
        return room, events

    room, events = run(go())
    assert [m.author for m in room.messages] == ["Human", "Good"]
    assert any(e["type"] == "error" and e["agent"] == "Bad" for e in events)


# ---- agent validation & websocket roster ----------------------------------


def test_agent_validation(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    existing = [Agent("Ada", "a", "")]
    with pytest.raises(AgentError, match="already"):
        agent_from_dict({"name": "ada", "provider": "ollama"}, existing)
    with pytest.raises(AgentError, match="Name"):
        agent_from_dict({"name": "1bad name", "provider": "ollama"}, existing)
    with pytest.raises(AgentError, match="Unknown provider"):
        agent_from_dict({"name": "Z", "provider": "skynet"}, existing)
    with pytest.raises(AgentError, match="GROQ_API_KEY"):
        agent_from_dict({"name": "Z", "provider": "groq"}, existing)
    a = agent_from_dict({"name": "Olly", "provider": "ollama", "color": "javascript:alert(1)"}, existing)
    assert a.model_id == "llama3.2" and a.color == "#6b7280" and a.role == "teammate"


def drain_until(ws, kind):
    while True:
        ev = ws.receive_json()
        if ev["type"] == kind:
            return ev


def test_websocket_add_and_remove_agents():
    app = create_app(MockBackend(delay=0))
    with TestClient(app) as client, client.websocket_connect("/ws/mix") as ws:
        ws.receive_json()  # history
        ws.send_json({"type": "add_agent", "agent": {"name": "Olly", "role": "critic", "provider": "ollama", "model": "qwen3"}})
        roster = drain_until(ws, "roster")
        olly = next(a for a in roster["agents"] if a["name"] == "Olly")
        assert olly["provider"] == "ollama" and olly["model"] == "qwen3"

        ws.send_json({"type": "add_agent", "agent": {"name": "Olly", "provider": "ollama"}})
        assert "already" in drain_until(ws, "agent_error")["text"]

        ws.send_json({"type": "say", "text": "@Olly what do you think?"})
        msg = drain_until(ws, "stream_end")["message"]
        assert msg["author"] == "Olly" and "mock qwen3" in msg["text"]
        drain_until(ws, "status")

        ws.send_json({"type": "remove_agent", "name": "Olly"})
        roster = drain_until(ws, "roster")
        assert "Olly" not in [a["name"] for a in roster["agents"]]


def test_providers_api():
    with TestClient(create_app(MockBackend(delay=0))) as client:
        providers = {p["id"]: p for p in client.get("/api/providers").json()}
        assert {"anthropic", "ollama", "groq", "gemini", "openrouter", "huggingface", "mistral", "custom"} <= set(providers)
        assert providers["ollama"]["configured"] is True
        assert client.get("/api/providers/ollama/models").json() == {"models": ["llama3.2"]}
        assert client.get("/api/providers/nope/models").status_code == 404
