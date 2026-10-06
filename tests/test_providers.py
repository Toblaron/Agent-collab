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
    backend = OpenAICompatBackend(client)
    backend.RETRY_DELAYS = (0, 0)
    bid = run(backend.bid(GROQ_AGENT, [GROQ_AGENT], []))
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


def test_auto_roster_gives_every_usable_provider_an_agent(monkeypatch):
    from agent_collab import server
    from agent_collab.server import ProviderCheck

    monkeypatch.setenv("AGENT_COLLAB_DEFAULT_PROVIDER", "auto")

    def fake(ids):
        return [ProviderCheck(p, p in ids, f"{p}-model" if p in ids else None, "") for p in server.AUTO_ORDER]

    monkeypatch.setattr(server, "check_providers", lambda refresh=False: fake({"gemini", "groq"}))
    team = server.default_roster()
    assert [(a.name, a.provider, a.model) for a in team] == [
        ("Ada", "gemini", "gemini-model"), ("Bo", "groq", "groq-model"),
        ("Cy", "gemini", "gemini-model"), ("Dee", "groq", "groq-model"),
    ]

    six = {"gemini", "groq", "openrouter", "mistral", "huggingface", "ollama"}
    monkeypatch.setattr(server, "check_providers", lambda refresh=False: fake(six))
    team = server.default_roster()
    assert len(team) == 6 and {a.provider for a in team} == six
    assert [a.name for a in team][4:] == ["Eve", "Fox"]

    monkeypatch.setattr(server, "check_providers", lambda refresh=False: fake(set()))
    assert {a.provider for a in server.default_roster()} == {"anthropic"}  # nothing usable: plain default


def test_pick_model_falls_back_to_a_listed_chat_model():
    from agent_collab.server import pick_model

    assert pick_model("groq", "llama-3.3-70b-versatile", []) == "llama-3.3-70b-versatile"  # no list: trust default
    assert pick_model("groq", "gone", ["whisper-large-v3", "llama-guard-4", "llama-4-scout"]) == "llama-4-scout"
    assert pick_model("openrouter", "gone:free", ["openai/gpt-x", "qwen/qwen3:free"]) == "qwen/qwen3:free"
    assert pick_model("gemini", "gone", ["gemini-embedding-001", "gemini-3-flash"]) == "gemini-3-flash"


def test_check_provider_reports_key_problems(monkeypatch):
    from agent_collab import server

    monkeypatch.setenv("GROQ_API_KEY", "bad")
    monkeypatch.setattr(server.httpx, "get", lambda *a, **k: httpx.Response(401, json={}))
    c = server.check_provider("groq")
    assert not c.ok and "key rejected" in c.note and "GROQ_API_KEY" in c.note

    monkeypatch.setattr(server.httpx, "get", lambda *a, **k: httpx.Response(200, json={"data": [{"id": "llama-3.3-70b-versatile"}]}))
    monkeypatch.setattr(server.httpx, "post", lambda *a, **k: httpx.Response(200, json={"choices": []}))
    assert server.check_provider("groq") == server.ProviderCheck("groq", True, "llama-3.3-70b-versatile", "ok")

    monkeypatch.delenv("GROQ_API_KEY")
    assert server.check_provider("groq").note == "no key"

    def refuse(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(server.httpx, "get", refuse)
    assert server.check_provider("ollama").note == "not running"


def test_gemini_bad_key_400_counts_as_rejected(monkeypatch):
    from agent_collab import server

    monkeypatch.setenv("GEMINI_API_KEY", "bad")
    monkeypatch.setattr(server.httpx, "get", lambda *a, **k: httpx.Response(400, text='{"error": "API key not valid"}'))
    assert "key rejected" in server.check_provider("gemini").note


def test_keys_env_loader_forgives_paste_mess(tmp_path, monkeypatch):
    from agent_collab.keysfile import load_keys, mask

    for k in ("GROQ_API_KEY", "GEMINI_API_KEY", "HF_TOKEN", "MISTRAL_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    f = tmp_path / "keys.env"
    f.write_text(
        "# comment\r\n"
        "GROQ_API_KEY= gsk_abc123def456\r\n"           # space after = (breaks `source`)
        'GEMINI_API_KEY="AIzaXYZ987654321"\n'          # quotes
        "export HF_TOKEN=​hf_tok en1234\n"   # export + invisible chars from a paste
        "MISTRAL_API_KEY=\n"                            # blank: ignored
        "OPENROUTER_KEY=sk-or-1\n"                      # typo in name
        "GEMINI_API_KEY_2\n",                           # no '='
        encoding="utf-8",
    )
    keys, problems = load_keys(f)
    assert keys == {"GROQ_API_KEY": "gsk_abc123def456", "GEMINI_API_KEY": "AIzaXYZ987654321", "HF_TOKEN": "hf_token1234"}
    import os
    assert os.environ["GROQ_API_KEY"] == "gsk_abc123def456" and "MISTRAL_API_KEY" not in os.environ
    assert any("did you mean OPENROUTER_API_KEY" in p for p in problems)
    assert any("no '='" in p for p in problems)
    assert mask("gsk_abc123def456") == "gsk_…f456 (16 chars)"


def test_keys_env_flags_key_on_wrong_line(tmp_path, monkeypatch):
    from agent_collab.keysfile import load_keys

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    f = tmp_path / "keys.env"
    f.write_text("GEMINI_API_KEY=gsk_groqkey123456\n")
    _, problems = load_keys(f)
    assert any("looks like a GROQ_API_KEY key" in p for p in problems)


def test_check_falls_back_past_retired_and_busy_models(monkeypatch):
    """The real Gemini situation on 2026-10-06: default retired (404), flash busy (503), lite works."""
    from agent_collab import server

    monkeypatch.setenv("GEMINI_API_KEY", "AQ.test")
    listed = ["gemini-2.5-flash", "gemini-embedding-001", "gemini-flash-latest", "gemini-flash-lite-latest"]
    monkeypatch.setattr(server.httpx, "get", lambda *a, **k: httpx.Response(200, json={"data": [{"id": f"models/{m}"} for m in listed]}))
    status = {"gemini-flash-latest": 503, "gemini-flash-lite-latest": 200, "gemini-2.5-flash": 404}
    probed = []

    def post(url, json, **k):
        probed.append(json["model"])
        return httpx.Response(status[json["model"]], text="{}")

    monkeypatch.setattr(server.httpx, "post", post)
    c = server.check_provider("gemini")
    assert (c.ok, c.model) == (True, "gemini-flash-lite-latest")
    assert probed == ["gemini-flash-latest", "gemini-flash-lite-latest"]  # default first, embeddings never probed

    status["gemini-flash-lite-latest"] = 503
    c = server.check_provider("gemini")
    assert (c.ok, c.model) == (True, "gemini-flash-latest") and "busy" in c.note  # busy beats nothing

    for m in status:
        status[m] = 404
    assert not server.check_provider("gemini").ok


def test_compat_backend_retries_busy_provider(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    calls = {"n": 0}

    def flaky(request):  # first call of each kind is "busy", then it works
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, json={"error": {"message": "high demand"}})
        if json.loads(request.content).get("stream"):
            return httpx.Response(200, text='data: {"choices": [{"delta": {"content": "hi"}}]}\n\ndata: [DONE]\n\n')
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"urgency": 0.5, "reason": "r"}'}}]})

    def make(handler):
        b = OpenAICompatBackend(httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        b.RETRY_DELAYS = (0, 0)
        return b

    async def speak(b):
        return "".join([t async for t in b.speak(GROQ_AGENT, [GROQ_AGENT], [])])

    backend = make(flaky)
    assert run(backend.bid(GROQ_AGENT, [GROQ_AGENT], [])).urgency == 0.5
    calls["n"] = 0
    assert run(speak(backend)) == "hi" and calls["n"] == 2

    busy = make(lambda r: httpx.Response(503, json={"error": {"message": "high demand"}}))
    assert run(busy.bid(GROQ_AGENT, [GROQ_AGENT], [])).reason == "(provider busy)"
    with pytest.raises(Exception, match="503: high demand"):
        run(speak(busy))
