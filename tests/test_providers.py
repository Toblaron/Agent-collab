import asyncio
import dataclasses
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from agent_collab.agents import Agent, AgentError, Message, agent_from_dict
from agent_collab.llm import Bid, MockBackend, OpenAICompatBackend, ProviderError, RoutingBackend, ThinkFilter, parse_bid
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
        ("Turing", "gemini", "gemini-model"), ("Tesla", "groq", "groq-model"),
        ("Socrates", "gemini", "gemini-model"), ("Curie", "groq", "openai/gpt-oss-120b"),
    ]  # Groq's second seat runs OpenAI's open model, so the team mixes model families

    six = {"gemini", "groq", "openrouter", "mistral", "huggingface", "ollama"}
    monkeypatch.setattr(server, "check_providers", lambda refresh=False: fake(six))
    team = server.default_roster()
    assert len(team) == 7 and {a.provider for a in team} == six  # no seat to share: gpt-oss gets a new one
    assert [(a.name, a.model) for a in team][4:] == [
        ("DaVinci", "huggingface-model"), ("Feynman", "ollama-model"), ("Franklin", "openai/gpt-oss-120b"),
    ]

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
    """Real Gemini behaviour seen on 2026-10-06: models listed but retired (404) or overloaded (503)."""
    from agent_collab import server

    monkeypatch.setenv("GEMINI_API_KEY", "AQ.test")
    listed = ["gemini-2.5-flash", "gemini-embedding-001", "gemini-flash-latest", "gemini-flash-lite-latest"]
    monkeypatch.setattr(server.httpx, "get", lambda *a, **k: httpx.Response(200, json={"data": [{"id": f"models/{m}"} for m in listed]}))
    status = {"gemini-flash-lite-latest": 503, "gemini-flash-latest": 200, "gemini-2.5-flash": 404}
    probed = []

    def post(url, json, **k):
        probed.append(json["model"])
        return httpx.Response(status[json["model"]], text="{}")

    monkeypatch.setattr(server.httpx, "post", post)
    c = server.check_provider("gemini")
    assert (c.ok, c.model) == (True, "gemini-flash-latest")
    assert probed == ["gemini-flash-lite-latest", "gemini-flash-latest"]  # default first, embeddings never probed

    status["gemini-flash-latest"] = 503
    c = server.check_provider("gemini")
    assert (c.ok, c.model) == (True, "gemini-flash-lite-latest") and "busy" in c.note  # busy beats nothing

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
    with pytest.raises(ProviderError, match="overloaded") as exc:
        run(speak(busy))
    assert exc.value.temporary


def test_provider_check_never_hangs_startup(monkeypatch):
    import time as _time

    from agent_collab import server

    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    def slow_or_fast(pid, budget=0):
        if pid == "groq":
            _time.sleep(5)  # a provider that hangs
        return server.ProviderCheck(pid, False, None, "no key")

    monkeypatch.setattr(server, "check_provider", slow_or_fast)
    seen = []
    start = _time.monotonic()
    checks = server.check_providers(refresh=True, deadline=0.5, on_result=seen.append)
    assert _time.monotonic() - start < 2
    groq = next(c for c in checks if c.id == "groq")
    assert groq.ok and "timed out" in groq.note  # key is trusted, real errors surface in the chat
    assert len(seen) == len(server.AUTO_ORDER)  # progress reported for every provider


def test_fast_start_skips_network(monkeypatch):
    from agent_collab import server

    monkeypatch.setenv("AGENT_COLLAB_SKIP_CHECK", "1")
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.setattr(server, "check_provider", lambda *a: (_ for _ in ()).throw(AssertionError("no network!")))
    checks = {c.id: c for c in server.check_providers(refresh=True)}
    assert checks["groq"].ok and checks["groq"].note == "not checked (fast start)"
    assert not checks["ollama"].ok


def test_quota_vs_rate_limit_classification():
    from agent_collab.llm import quota_used_up

    daily = httpx.Response(429, text='{"error": {"message": "You exceeded your current quota... free_tier_requests, limit: 20"}}')
    minute = httpx.Response(429, text='{"error": {"message": "Quota exceeded for metric x. Please retry in 41.2s."}}')
    plain = httpx.Response(429, text='{"error": {"message": "Too many requests"}}')
    assert quota_used_up(daily) and not quota_used_up(minute) and not quota_used_up(plain)
    assert not quota_used_up(httpx.Response(503, text="quota"))


def test_daily_quota_is_reported_not_retried(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(429, json={"error": {"message": "You exceeded your current quota (limit: 20)"}})

    agent = Agent("Gem", "g", "", provider="gemini", model="gemini-x")
    backend = OpenAICompatBackend(httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    bid = run(backend.bid(agent, [agent], []))
    assert bid.reason == "(daily quota used up)" and calls["n"] == 1  # no pointless retry

    async def speak():
        return "".join([t async for t in backend.speak(agent, [agent], [])])

    with pytest.raises(ProviderError, match="limit for gemini-x is reached") as exc:
        run(speak())
    assert exc.value.model_unavailable


def test_check_skips_models_with_used_up_quota(monkeypatch):
    from agent_collab import server

    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setattr(server.httpx, "get", lambda *a, **k: httpx.Response(200, json={"data": [
        {"id": "models/gemini-flash-lite-latest"}, {"id": "models/gemini-flash-latest"}]}))

    def post(url, json, **k):
        if json["model"] == "gemini-flash-lite-latest":
            return httpx.Response(429, text='{"error":{"message":"You exceeded your current quota"}}')
        return httpx.Response(200, json={"choices": []})

    monkeypatch.setattr(server.httpx, "post", post)
    c = server.check_provider("gemini")
    assert (c.ok, c.model) == (True, "gemini-flash-latest") and "quota used up" in c.note


def test_pacer_spaces_requests_and_adapts():
    from agent_collab.llm import Pacer

    async def go():
        p = Pacer(base=0.05)
        loop = asyncio.get_running_loop()
        start = loop.time()
        for _ in range(4):
            await p.wait()
        spaced = loop.time() - start
        p.slow_down()
        slowed = p.interval
        for _ in range(10):
            p.ok()
        return spaced, slowed, p.interval

    spaced, slowed, recovered = run(go())
    assert spaced >= 0.15  # 4 requests, 3 gaps of 0.05s
    assert slowed == 1.0   # a 429 pushes the gap to at least 1s
    assert 0.05 <= recovered < 0.2  # and successes ease it back toward the provider's rate


def test_retry_after_header_is_respected(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "k")
    times = []

    def handler(request):
        times.append(asyncio.get_event_loop().time())
        if len(times) == 1:
            return httpx.Response(429, headers={"retry-after": "0.3"}, json={"message": "Rate limit exceeded"})
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"urgency": 0.4, "reason": "r"}'}}]})

    agent = Agent("Dee", "researcher", "", provider="mistral")
    backend = OpenAICompatBackend(httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    backend.RETRY_DELAYS = (0.01, 0.01)
    assert run(backend.bid(agent, [agent], [])).urgency == 0.4
    assert times[1] - times[0] >= 0.29  # waited what the provider asked, not just our short delay


def test_mistral_is_paced_by_default():
    backend = OpenAICompatBackend()
    assert backend.pacer("mistral").interval == 1.1 and backend.pacer("groq").interval == 0


def test_room_on_strict_one_request_per_window_provider(monkeypatch):
    """Reproduces the real failure: Mistral's free tier rejects requests that come too close
    together, and Dee (on Mistral) was asked to open the debate."""
    from agent_collab import llm
    from agent_collab.room import Room, RoomSettings

    window = 0.2
    monkeypatch.setitem(llm.BASE_INTERVALS, "mistral", window + 0.05)
    monkeypatch.setenv("MISTRAL_API_KEY", "k")
    last = {"t": -10.0}
    rejected = {"n": 0}

    def strict(request):
        now = asyncio.get_event_loop().time()
        too_soon = now - last["t"] < window
        last["t"] = now
        if too_soon:
            rejected["n"] += 1
            return httpx.Response(429, json={"object": "error", "message": "Rate limit exceeded", "code": "1300"})
        body = json.loads(request.content)
        if body.get("stream"):
            return httpx.Response(200, text='data: {"choices": [{"delta": {"content": "Bostrom says..."}}]}\n\ndata: [DONE]\n\n')
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"urgency": 0.2, "reason": "r"}'}}]})

    dee = Agent("Dee", "researcher", "", provider="mistral", tools=())
    cy = Agent("Cy", "critic", "", provider="mistral", tools=())
    backend = OpenAICompatBackend(httpx.AsyncClient(transport=httpx.MockTransport(strict)))
    backend.RETRY_DELAYS = (0.3, 0.6)

    async def go():
        room = Room("sim", [dee, cy], backend, RoomSettings(max_agent_turns=3, speak_retry_wait=0.3))
        q = room.subscribe()
        await room.post_human("Topic: are we in a simulation? @Dee open.")
        await room.wait_idle()
        return room, [q.get_nowait() for _ in range(q.qsize())]

    room, events = run(go())
    assert room.messages[1].author == "Dee" and "Bostrom" in room.messages[1].text
    assert not [e for e in events if e["type"] == "error"], [e for e in events if e["type"] == "error"]
    assert rejected["n"] == 0  # paced from the start: never even tripped the limit


GROQ_TPM = ("Rate limit reached for model `qwen/qwen3.8-27b` in organization `org_x` service tier `on_demand` on tokens "
            "per minute (TPM): Limit 6000, Used 5800, Requested 900. Please try again in 5.12s. Need more tokens? "
            "Upgrade to Dev Tier today at https://console.groq.com/settings/billing")
GROQ_TPD = ("Rate limit reached for model `qwen/qwen3.8-27b` in organization `org_x` service tier `on_demand` on tokens "
            "per day (TPD): Limit 500000, Used 499800, Requested 900. Please try again in 7m12.5s. Need more tokens? "
            "Upgrade to Dev Tier today at https://console.groq.com/settings/billing")


def groq_429(msg, **headers):
    return httpx.Response(429, headers=headers, json={"error": {"message": msg, "type": "tokens", "code": "rate_limit_exceeded"}})


def test_wait_times_are_parsed_from_real_messages():
    from agent_collab.llm import limit_wait, quota_used_up, unavailable_reason

    assert limit_wait(groq_429(GROQ_TPM)) == pytest.approx(5.12)
    assert limit_wait(groq_429(GROQ_TPD)) == pytest.approx(432.5)
    assert limit_wait(groq_429("slow down, try again in 450ms")) == pytest.approx(0.45)
    assert limit_wait(groq_429("x", **{"retry-after": "30"})) == 30
    assert limit_wait(groq_429("Please retry in 41.2s.")) == pytest.approx(41.2)
    # The bug: per-minute limits mention "billing" and were treated as a used-up quota.
    assert not quota_used_up(groq_429(GROQ_TPM))
    assert quota_used_up(groq_429(GROQ_TPD))
    assert unavailable_reason(groq_429(GROQ_TPD), "Groq", "qwen/qwen3.8-27b") == \
        "Groq's daily limit for qwen/qwen3.8-27b is reached (resets in ~7 min)"


def test_alternative_model_skips_blocked_and_non_chat_models(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    listed = ["whisper-large-v3", "llama-guard-4-12b", "qwen/qwen3.8-27b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"]

    def handler(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": m} for m in listed]})
        return groq_429(GROQ_TPD)

    backend = OpenAICompatBackend(httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    bo = Agent("Bo", "builder", "", provider="groq", model="qwen/qwen3.8-27b")

    async def go():
        bid = await backend.bid(bo, [bo], [])   # trips the daily limit -> qwen blocked
        alt = await backend.alternative_model(bo)
        backend._blocked[("groq", alt)] = asyncio.get_running_loop().time() + 999
        alt2 = await backend.alternative_model(dataclasses.replace(bo, model=alt))
        return bid, alt, alt2

    bid, alt, alt2 = run(go())
    assert bid.reason == "(daily quota used up)"
    assert alt == "llama-3.3-70b-versatile"          # Groq's default comes first
    assert alt2 == "llama-3.1-8b-instant"            # never whisper/guard, never a blocked model


def test_agent_switches_model_mid_conversation_and_still_answers(monkeypatch):
    """The reported failure: Bo on Groq/qwen hit its daily limit. Now he moves to another
    Groq model, the room says so, and he finishes his turn."""
    from agent_collab.room import Room, RoomSettings

    monkeypatch.setenv("GROQ_API_KEY", "k")
    used = []

    def handler(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "qwen/qwen3.8-27b"}, {"id": "llama-3.3-70b-versatile"}]})
        body = json.loads(request.content)
        used.append(body["model"])
        if body["model"] == "qwen/qwen3.8-27b":
            return groq_429(GROQ_TPD)
        if body.get("stream"):
            return httpx.Response(200, text='data: {"choices": [{"delta": {"content": "Here is the build plan."}}]}\n\ndata: [DONE]\n\n')
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"urgency": 0.1, "reason": "r"}'}}]})

    bo = Agent("Bo", "builder", "", provider="groq", model="qwen/qwen3.8-27b", tools=())
    backend = OpenAICompatBackend(httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    async def go():
        room = Room("r", [bo], backend, RoomSettings(max_agent_turns=1))
        q = room.subscribe()
        await room.post_human("@Bo build it")
        await room.wait_idle()
        return room, [q.get_nowait() for _ in range(q.qsize())]

    room, events = run(go())
    assert room.messages[-1].author == "Bo" and room.messages[-1].text == "Here is the build plan."
    assert room.agents[0].model == "llama-3.3-70b-versatile"  # the switch sticks for later turns
    assert room.messages[-1].meta["model"] == "llama-3.3-70b-versatile"
    notice = next(e["text"] for e in events if e["type"] == "notice" and "switched" in e["text"])
    assert notice.startswith("Bo switched to llama-3.3-70b-versatile: Groq's daily limit for qwen/qwen3.8-27b is reached")
    assert not [e for e in events if e["type"] == "error"]
    assert used == ["qwen/qwen3.8-27b", "llama-3.3-70b-versatile"]  # no retry storm against the dead model


def test_per_minute_limit_is_paced_not_switched(monkeypatch):
    from agent_collab.room import Room, RoomSettings

    monkeypatch.setenv("GROQ_API_KEY", "k")
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return groq_429(GROQ_TPM.replace("5.12s", "0.2s"))
        return httpx.Response(200, text='data: {"choices": [{"delta": {"content": "ok"}}]}\n\ndata: [DONE]\n\n')

    bo = Agent("Bo", "builder", "", provider="groq", model="qwen/qwen3.8-27b", tools=())
    backend = OpenAICompatBackend(httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    backend.RETRY_DELAYS = (0.05, 0.05)

    async def go():
        room = Room("r", [bo], backend, RoomSettings(max_agent_turns=1))
        await room.post_human("@Bo go")
        await room.wait_idle()
        return room

    room = run(go())
    assert room.agents[0].model == "qwen/qwen3.8-27b" and room.messages[-1].text == "ok"


def test_openai_reasoning_models_get_their_own_parameters():
    from agent_collab.providers import adapt_body

    body = {"model": "gpt-5-mini", "messages": [], "temperature": 0.7, "max_tokens": 2048, "stream": True}
    out = adapt_body("openai", body)
    assert "max_tokens" not in out and "temperature" not in out
    assert out["reasoning_effort"] == "low"
    assert out["max_completion_tokens"] > 2048  # room for hidden reasoning plus the reply
    assert body["max_tokens"] == 2048  # caller's dict untouched

    classic = adapt_body("openai", {"model": "gpt-4.1-mini", "temperature": 0.2, "max_tokens": 5})
    assert classic == {"model": "gpt-4.1-mini", "temperature": 0.2, "max_completion_tokens": 5}
    # gpt-oss on Groq is a normal OpenAI-compatible chat model: body passes through as-is
    groq = {"model": "openai/gpt-oss-120b", "temperature": 0.2, "max_tokens": 5}
    assert adapt_body("groq", groq) is groq


def test_openai_model_list_skips_non_chat_models():
    from agent_collab.providers import candidate_models, pick_model

    ids = ["dall-e-3", "gpt-4o-mini", "gpt-4o-mini-search-preview", "gpt-4o-realtime-preview",
           "gpt-5-mini", "gpt-image-1", "omni-moderation-latest", "text-embedding-3-small", "whisper-1"]
    assert pick_model("openai", "gpt-9-mini", ids) == "gpt-5-mini"
    assert candidate_models("openai", "gpt-5-mini", ids)[:2] == ["gpt-5-mini", "gpt-4o-mini"]
    # Groq also hosts OpenAI's free open-weight models; they're a fallback when Llama's quota runs out
    assert "openai/gpt-oss-120b" in candidate_models("groq", "llama-3.3-70b-versatile",
                                                    ["llama-3.3-70b-versatile", "openai/gpt-oss-120b"])


def test_gpt_oss_seat_follows_what_the_host_lists(monkeypatch):
    from agent_collab import server
    from agent_collab.server import ProviderCheck

    monkeypatch.setenv("AGENT_COLLAB_DEFAULT_PROVIDER", "auto")
    no_oss = [ProviderCheck("gemini", True, "g", ""), ProviderCheck("groq", True, "llama", "", ("llama",))]
    monkeypatch.setattr(server, "check_providers", lambda refresh=False: no_oss)
    assert not any("gpt-oss" in a.model_id for a in server.default_roster())  # Groq doesn't list it

    via_openrouter = [ProviderCheck("gemini", True, "g", ""),
                      ProviderCheck("openrouter", True, "x:free", "", ("openai/gpt-oss-120b:free", "x:free"))]
    monkeypatch.setattr(server, "check_providers", lambda refresh=False: via_openrouter)
    team = server.default_roster()
    assert [a.model_id for a in team].count("openai/gpt-oss-120b:free") == 1

    monkeypatch.setenv("AGENT_COLLAB_DEFAULT_PROVIDER", "groq")
    team = server.default_roster()
    assert [a.model_id for a in team] == ["llama-3.3-70b-versatile"] * 3 + ["openai/gpt-oss-120b"]
    monkeypatch.setenv("AGENT_COLLAB_DEFAULT_MODEL", "qwen-x")  # an explicit model wins for everyone
    assert {a.model_id for a in server.default_roster()} == {"qwen-x"}


def test_openrouter_prefers_known_free_models_over_alphabetical():
    from agent_collab.providers import candidate_models, pick_model

    ids = ["apodex/apodex-1.1-mini:free", "deepseek/deepseek-chat", "deepseek/deepseek-chat-v3.1:free",
           "openai/gpt-oss-120b:free", "qwen/qwen3-235b:free", "zz/obscure:free"]
    assert pick_model("openrouter", "meta-llama/llama-3.3-70b-instruct:free", ids) == "deepseek/deepseek-chat-v3.1:free"
    cands = candidate_models("openrouter", "meta-llama/llama-3.3-70b-instruct:free", ids)
    assert "deepseek/deepseek-chat" not in cands  # paid variant never offered
    assert cands[:3] == ["deepseek/deepseek-chat-v3.1:free", "qwen/qwen3-235b:free", "openai/gpt-oss-120b:free"]
    assert cands[-1] == "zz/obscure:free"  # unknown free models still count, just last


def test_check_note_names_the_retired_default(monkeypatch):
    import httpx

    from agent_collab import server

    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setattr(httpx, "get", lambda *a, **k: httpx.Response(200, json={"data": [{"id": "openai/gpt-oss-120b"}]}))
    monkeypatch.setattr(server, "probe_chat", lambda spec, model, timeout=12.0: (200, "{}"))
    check = server.check_provider("groq")
    assert check.ok and check.model == "openai/gpt-oss-120b"
    assert check.note == "ok (using openai/gpt-oss-120b; llama-3.3-70b-versatile is no longer offered)"
