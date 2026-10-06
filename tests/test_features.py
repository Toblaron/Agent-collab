import asyncio

import pytest
from fastapi.testclient import TestClient

from agent_collab.agents import Agent, AgentError, Message, render_transcript
from agent_collab.llm import Bid, MockBackend
from agent_collab.room import Room, RoomSettings
from agent_collab.rooms import RoomStore, export_markdown
from agent_collab.server import create_app
from agent_collab.teams import TeamStore

A, B, C = Agent("A", "a", ""), Agent("B", "b", ""), Agent("C", "c", "")


def run(coro):
    return asyncio.run(coro)


class Scripted:
    """urgency per agent name; speech per agent name (default 'hello from X')."""

    def __init__(self, urgency=None, speech=None):
        self.urgency = urgency or {}
        self.speech = speech or {}
        self.spoke = []
        self.bidders = []

    async def bid(self, agent, roster, transcript, whiteboard=""):
        self.bidders.append(agent.name)
        u = self.urgency.get(agent.name, 0.0)
        return Bid(urgency=u(transcript) if callable(u) else u, reason="")

    async def speak(self, agent, roster, transcript, whiteboard=""):
        self.spoke.append(agent.name)
        text = self.speech.get(agent.name, f"hello from {agent.name}")
        if text:
            yield text


def drain(ws, kind):
    while True:
        ev = ws.receive_json()
        if ev["type"] == kind:
            return ev


def make_app(tmp_path, backend=None):
    return create_app(
        backend or MockBackend(delay=0), searcher=None,
        teams=TeamStore(tmp_path), room_store=RoomStore(tmp_path),
    )


# ---- persistence ----------------------------------------------------------------


def test_room_survives_a_server_restart(tmp_path):
    with TestClient(make_app(tmp_path)) as client, client.websocket_connect("/ws/trip") as ws:
        ws.receive_json()
        ws.send_json({"type": "say", "text": "@Ada plan a trip"})
        drain(ws, "stream_end")
        ws.send_json({"type": "set_whiteboard", "text": "# Trip plan"})
        drain(ws, "whiteboard")
        ws.send_json({"type": "set_muted", "name": "Cy", "muted": True})
        drain(ws, "roster")
        ws.send_json({"type": "set_max_turns", "value": 20})
        drain(ws, "settings")
        ws.send_json({"type": "stop"})

    # A brand-new app instance = a restart; it must find everything on disk.
    with TestClient(make_app(tmp_path)) as client, client.websocket_connect("/ws/trip") as ws:
        state = ws.receive_json()
        assert state["messages"][0]["text"] == "@Ada plan a trip"
        assert state["messages"][1]["author"] == "Ada"
        assert state["messages"][1]["meta"]["model"]  # who answered, with which model
        assert state["whiteboard"] == "# Trip plan" and state["whiteboard_by"] == "Human"
        assert state["muted"] == ["Cy"] and state["max_turns"] == 20
        assert state["title"] == "@Ada plan a trip"
        rooms = client.get("/api/rooms").json()
        assert rooms[0]["id"] == "trip" and rooms[0]["messages"] >= 2


def test_empty_rooms_are_not_saved(tmp_path):
    with TestClient(make_app(tmp_path)) as client, client.websocket_connect("/ws/ghost") as ws:
        ws.receive_json()
        assert client.get("/api/rooms").json() == []
    assert not (tmp_path / "rooms" / "ghost.json").exists()


def test_corrupt_room_file_is_set_aside(tmp_path):
    store = RoomStore(tmp_path)
    store.dir.mkdir(parents=True)
    (store.dir / "bad.json").write_text("{nope")
    assert store.load("bad") is None
    assert not (store.dir / "bad.json").exists()
    assert len(list(store.dir.glob("bad.corrupt-*.json"))) == 1
    assert store.list() == []


def test_export_and_delete(tmp_path):
    with TestClient(make_app(tmp_path)) as client:
        with client.websocket_connect("/ws/exp") as ws:
            ws.receive_json()
            ws.send_json({"type": "say", "text": "@Bo write a haiku"})
            drain(ws, "stream_end")
            ws.send_json({"type": "stop"})
        r = client.get("/api/rooms/exp/export.md")
        assert r.status_code == 200 and "attachment" in r.headers["content-disposition"]
        assert "# @Bo write a haiku" in r.text and "**Bo**" in r.text and "## Conversation" in r.text
        assert client.get("/api/rooms/nope/export.md").status_code == 404
        assert client.get("/api/rooms/..%2Fetc/export.md").status_code in (400, 404)
        assert client.delete("/api/rooms/exp").json() == {"deleted": True}
        assert client.get("/api/rooms").json() == []


def test_export_markdown_shape():
    md = export_markdown({
        "id": "x", "title": "T", "agents": [{"name": "Ada", "role": "architect", "model": "m"}],
        "whiteboard": "# Plan", "whiteboard_by": "Ada",
        "messages": [{"author": "Human", "text": "hi", "ts": 0}, {"author": "Ada", "text": "yo", "ts": 0, "meta": {"model": "m", "secs": 1.5}}],
    })
    assert "**Team:** Ada (architect, m)" in md and "## Whiteboard (last edit: Ada)" in md and "· m · 1.5s" in md


# ---- room controls ----------------------------------------------------------------


def test_muted_agent_neither_bids_nor_takes_mentions():
    async def go():
        backend = Scripted(urgency={"A": 0.9, "B": 0.9})
        room = Room("r", [A, B], backend, RoomSettings(max_agent_turns=2, dominance_penalty=0))
        room.set_muted("a", True)  # case-insensitive
        await room.post_human("@A are you there?")
        await room.wait_idle()
        return backend

    backend = run(go())
    assert "A" not in backend.spoke and "A" not in backend.bidders and backend.spoke[0] == "B"


def test_continue_lets_the_team_pick_up_again():
    async def go():
        quiet = {"on": True}
        backend = Scripted(urgency={"A": lambda t: 0.0 if quiet["on"] else 0.8})
        room = Room("r", [A, B], backend, RoomSettings(max_agent_turns=1))
        with pytest.raises(AgentError):
            await room.continue_conversation()  # nothing to continue yet
        await room.post_human("hi")
        await room.wait_idle()
        assert backend.spoke == []
        quiet["on"] = False
        await room.continue_conversation()
        await room.wait_idle()
        return backend

    assert run(go()).spoke == ["A"]


def test_clear_keeps_team_and_wipes_chat():
    async def go():
        room = Room("r", [A, B], Scripted(urgency={"A": 0.9}), RoomSettings(max_agent_turns=1))
        await room.post_human("hi")
        await room.wait_idle()
        room.set_whiteboard("x", by="Human")
        await room.clear()
        return room

    room = run(go())
    assert room.messages == [] and room.whiteboard == "" and [a.name for a in room.agents] == ["A", "B"]


def test_update_agent_in_place(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    room = Room("r", [A, B], Scripted())
    room.set_muted("B", True)
    room.update_agent("B", {"name": "Bea", "provider": "groq", "model": "llama-x", "role": "tester"})
    bea = room.agents[1]
    assert (bea.name, bea.provider, bea.model, bea.role) == ("Bea", "groq", "llama-x", "tester")
    assert room.muted == {"Bea"}
    with pytest.raises(AgentError, match="already"):
        room.update_agent("Bea", {"name": "A"})
    with pytest.raises(AgentError, match="No agent"):
        room.update_agent("Zed", {})


def test_turn_limit_is_clamped_and_announced():
    async def go():
        room = Room("r", [A, B], Scripted(urgency={"A": 0.9, "B": 0.9}), RoomSettings(dominance_penalty=0))
        assert room.set_max_turns("999") == 50 and room.set_max_turns(-3) == 1 and room.set_max_turns("x") == 12
        room.set_max_turns(3)
        q = room.subscribe()
        await room.post_human("go")
        await room.wait_idle()
        return [e for e in [q.get_nowait() for _ in range(q.qsize())] if e["type"] == "notice"]

    notices = run(go())
    assert any("turn limit reached (3)" in n["text"] for n in notices)


# ---- bug fixes --------------------------------------------------------------------


def test_empty_reply_does_not_loop_forever():
    async def go():
        backend = Scripted(urgency={"A": 0.9, "B": 0.6}, speech={"A": ""})
        room = Room("r", [A, B], backend, RoomSettings(max_agent_turns=6, dominance_penalty=0))
        await room.post_human("hi")
        await room.wait_idle()
        return backend

    backend = run(go())
    assert backend.spoke.count("A") == 1  # benched after saying nothing
    assert "B" in backend.spoke


def test_backend_crash_in_bid_is_contained():
    class Exploding(Scripted):
        async def bid(self, agent, roster, transcript, whiteboard=""):
            if agent.name == "A":
                raise ZeroDivisionError("boom")
            return Bid(urgency=0.8, reason="")

    async def go():
        backend = Exploding()
        room = Room("r", [A, B], backend, RoomSettings(max_agent_turns=1))
        await room.post_human("hi")
        await room.wait_idle()
        return backend

    assert run(go()).spoke == ["B"]


def test_search_crash_is_contained():
    class BadSearcher:
        name = "bad"

        async def search(self, q):
            raise KeyError("layout changed")

    async def go():
        room = Room("r", [A], Scripted(urgency={"A": 0.9}, speech={"A": "[[search: x]]"}),
                    RoomSettings(max_agent_turns=1), searcher=BadSearcher())
        await room.post_human("hi")
        await room.wait_idle()
        return room

    room = run(go())
    assert "failed unexpectedly (KeyError)" in room.messages[-1].text


def test_websocket_rejects_garbage_without_dropping(tmp_path):
    with TestClient(make_app(tmp_path)) as client, client.websocket_connect("/ws/junk") as ws:
        ws.receive_json()
        ws.send_text("not json")
        assert "valid JSON" in drain(ws, "agent_error")["text"]
        ws.send_json([1, 2])
        assert "JSON objects" in drain(ws, "agent_error")["text"]
        ws.send_json({"type": "teleport"})
        assert "Unknown request" in drain(ws, "agent_error")["text"]
        ws.send_json({"type": "say", "text": "x" * 20_001})
        assert "too long" in drain(ws, "agent_error")["text"]
        ws.send_json({"type": "set_max_turns", "value": 5})  # still alive
        assert drain(ws, "settings")["max_turns"] == 5


def test_invalid_room_id_is_refused(tmp_path):
    from starlette.websockets import WebSocketDisconnect

    with TestClient(make_app(tmp_path)) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws/" + "x" * 41) as ws:
                ws.receive_json()


def test_transcript_window_keeps_prompts_small():
    msgs = [Message("Human", f"m{i}") for i in range(100)] + [Message("Ada", "y" * 9000)]
    text = render_transcript(msgs, window=10)
    assert "91 earlier messages not shown" in text
    assert "[Human]: m90" not in text and "[Human]: m91" in text
    assert text.endswith("…(truncated)") and len(text) < 6000


def test_permanent_failures_do_not_trigger_rate_limit_waits():
    class BadKeys(Scripted):
        async def bid(self, agent, roster, transcript, whiteboard=""):
            return Bid(urgency=0.0, reason="(error: key rejected)")

    async def go():
        room = Room("r", [A, B], BadKeys(), RoomSettings(max_agent_turns=3, throttle_waits=(30.0,)))
        q = room.subscribe()
        start = asyncio.get_event_loop().time()
        await room.post_human("hi")
        await room.wait_idle()
        events = [q.get_nowait() for _ in range(q.qsize())]
        return events, asyncio.get_event_loop().time() - start

    events, elapsed = run(go())
    assert elapsed < 1  # did not sit through the 30s rate-limit wait
    err = next(e for e in events if e["type"] == "error")
    assert "key rejected" in err["text"] and "doctor" in err["text"]


def test_snapshot_includes_reply_in_progress():
    class Slow(Scripted):
        async def speak(self, agent, roster, transcript, whiteboard=""):
            yield "half a "
            await asyncio.sleep(0.2)
            yield "thought"

    async def go():
        room = Room("r", [A], Slow(urgency={"A": 0.9}), RoomSettings(max_agent_turns=1))
        await room.post_human("hi")
        await asyncio.sleep(0.05)
        mid = room.snapshot()["live"]
        await room.wait_idle()
        return mid, room.snapshot()["live"]

    mid, after = run(go())
    assert mid["author"] == "A" and mid["text"] == "half a " and after is None


def test_agents_are_not_told_about_search_when_it_is_off():
    seen = []

    class Recorder(Scripted):
        async def speak(self, agent, roster, transcript, whiteboard=""):
            seen.append(agent.system_prompt(roster))
            yield "ok"

    async def go(searcher):
        room = Room("r", [A], Recorder(urgency={"A": 0.9}), RoomSettings(max_agent_turns=1), searcher=searcher)
        await room.post_human("hi")
        await room.wait_idle()

    class S:
        name = "s"

        async def search(self, q):
            return []

    run(go(None))
    run(go(S()))
    assert "[[search:" not in seen[0] and "```whiteboard" in seen[0]
    assert "[[search:" in seen[1]


def test_rate_limited_speaker_gets_a_second_try():
    from agent_collab.llm import ProviderError

    class Flaky(Scripted):
        tries = 0

        async def speak(self, agent, roster, transcript, whiteboard=""):
            Flaky.tries += 1
            if Flaky.tries == 1:
                raise ProviderError("Mistral is rate-limiting requests right now", temporary=True)
            yield "here's my opening"

    async def go():
        room = Room("r", [A, B], Flaky(), RoomSettings(max_agent_turns=1, speak_retry_wait=0.01))
        q = room.subscribe()
        await room.post_human("@A open the debate")
        await room.wait_idle()
        return room, [q.get_nowait() for _ in range(q.qsize())]

    room, events = run(go())
    assert room.messages[-1].author == "A" and room.messages[-1].text == "here's my opening"
    assert any(e["type"] == "notice" and "trying again" in e["text"] for e in events)
    assert not any(e["type"] == "error" for e in events)


def test_permanent_speak_error_is_not_retried():
    from agent_collab.llm import ProviderError

    class Broken(Scripted):
        tries = 0

        async def speak(self, agent, roster, transcript, whiteboard=""):
            Broken.tries += 1
            raise ProviderError("Mistral returned 404: model not found")
            yield

    async def go():
        room = Room("r", [A], Broken(), RoomSettings(max_agent_turns=1, speak_retry_wait=5))
        q = room.subscribe()
        await room.post_human("@A go")
        await room.wait_idle()
        return [q.get_nowait() for _ in range(q.qsize())]

    events = run(go())
    assert Broken.tries == 1
    err = next(e for e in events if e["type"] == "error")
    assert err["text"] == "Mistral returned 404: model not found"  # no "ProviderError:" noise
