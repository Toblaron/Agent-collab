import asyncio

import pytest
from fastapi.testclient import TestClient

from agent_collab.agents import DEFAULT_ROSTER, Agent, Message
from agent_collab.llm import Bid, MockBackend
from agent_collab.room import HUMAN, Room, RoomSettings
from agent_collab.server import create_app


class ScriptedBackend:
    """Bids come from a fixed table; speech is a fixed string."""

    def __init__(self, urgencies: dict[str, float], delay: float = 0.0):
        self.urgencies = urgencies
        self.delay = delay
        self.spoke: list[str] = []

    async def bid(self, agent, roster, transcript, whiteboard=""):
        return Bid(urgency=self.urgencies.get(agent.name, 0.0), reason="scripted")

    async def speak(self, agent, roster, transcript, whiteboard=""):
        self.spoke.append(agent.name)
        for word in ["hello", "from", agent.name]:
            await asyncio.sleep(self.delay)
            yield word + " "


A = Agent("A", "a", "")
B = Agent("B", "b", "")


def run(coro):
    return asyncio.run(coro)


def test_highest_bid_speaks_and_room_goes_quiet():
    async def go():
        backend = ScriptedBackend({"A": 0.9, "B": 0.1})
        room = Room("r", [A, B], backend, RoomSettings(max_agent_turns=5))
        await room.post_human("hi")
        await room.wait_idle()
        return backend, room

    backend, room = run(go())
    # A speaks; then A cannot bid (just spoke) and B is below threshold -> silence.
    assert backend.spoke == ["A"]
    assert [m.author for m in room.messages] == [HUMAN, "A"]
    assert room.messages[-1].text == "hello from A"


def test_last_speaker_does_not_bid_so_agents_alternate():
    async def go():
        backend = ScriptedBackend({"A": 0.9, "B": 0.8})
        room = Room("r", [A, B], backend, RoomSettings(max_agent_turns=4, dominance_penalty=0.0))
        await room.post_human("hi")
        await room.wait_idle()
        return backend

    assert run(go()).spoke == ["A", "B", "A", "B"]


def test_dominance_penalty_lowers_score():
    room = Room("r", [A, B], ScriptedBackend({}), RoomSettings(dominance_penalty=0.2, recent_window=4))
    room.messages = [Message("A", "x"), Message("B", "y"), Message("A", "z")]
    assert room._score(A, Bid(urgency=0.5, reason="")) < 0.5 - 0.39
    assert room._score(B, Bid(urgency=0.5, reason="")) < 0.5 - 0.19


def test_human_message_interrupts_current_speaker():
    async def go():
        backend = ScriptedBackend({"A": 0.9}, delay=0.05)
        room = Room("r", [A, B], backend, RoomSettings(max_agent_turns=1))
        await room.post_human("first")
        await asyncio.sleep(0.07)  # A is mid-sentence
        await room.post_human("second")
        await room.wait_idle()
        return room

    room = run(go())
    texts = [(m.author, m.text) for m in room.messages]
    assert texts[0] == (HUMAN, "first")
    assert texts[1][0] == "A" and texts[1][1].endswith("(interrupted)")
    assert texts[2] == (HUMAN, "second")
    assert texts[3] == ("A", "hello from A")


def test_max_turns_caps_runaway_conversation():
    async def go():
        backend = ScriptedBackend({"A": 1.0, "B": 1.0})
        room = Room("r", [A, B], backend, RoomSettings(max_agent_turns=3, dominance_penalty=0.0))
        await room.post_human("go")
        await room.wait_idle()
        return backend

    assert len(run(go()).spoke) == 3


def test_mock_backend_conversation_terminates():
    async def go():
        room = Room("r", list(DEFAULT_ROSTER), MockBackend(delay=0))
        await room.post_human("plan a launch @Ada")
        await room.wait_idle()
        return room

    room = run(go())
    assert room.messages[1].author == "Ada"  # mention wins the floor
    assert 2 <= len(room.messages) <= 1 + RoomSettings().max_agent_turns


def test_websocket_end_to_end():
    app = create_app(MockBackend(delay=0))
    with TestClient(app) as client, client.websocket_connect("/ws/test") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "history" and len(hello["agents"]) == 4
        ws.send_json({"type": "say", "text": "@Bo build it"})
        seen = []
        while True:
            ev = ws.receive_json()
            seen.append(ev["type"])
            if ev["type"] == "status" and ev["state"] == "idle":
                break
        assert "message" in seen and "bids" in seen and "stream_delta" in seen and "stream_end" in seen


@pytest.mark.parametrize("path", ["/", "/healthz"])
def test_http_routes(path):
    with TestClient(create_app(MockBackend(delay=0))) as client:
        assert client.get(path).status_code == 200
