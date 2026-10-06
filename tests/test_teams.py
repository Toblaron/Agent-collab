import pytest
from fastapi.testclient import TestClient

from agent_collab.agents import Agent
from agent_collab.llm import MockBackend
from agent_collab.server import create_app
from agent_collab.teams import TeamError, TeamStore

TEAM = [Agent("Olly", "critic", "Picky.", "#00ff41", provider="ollama", model="qwen3", tools=("search",))]


def test_save_list_load_delete(tmp_path):
    store = TeamStore(tmp_path)
    store.save("Night Shift", TEAM)
    assert [t["name"] for t in store.list()] == ["Night Shift"]
    assert store.list()[0]["agents"] == ["Olly"]
    loaded = store.load("night shift")  # names are case-insensitive via the slug
    assert loaded[0]["model"] == "qwen3" and loaded[0]["tools"] == ["search"]
    store.save("Night Shift", TEAM)  # overwrite, not duplicate
    assert len(store.list()) == 1
    store.delete("Night Shift")
    assert store.list() == []


@pytest.mark.parametrize("name", ["../../etc/passwd", "a/b", ".hidden", "", "x" * 41, "nul\x00"])
def test_bad_team_names_are_rejected(tmp_path, name):
    with pytest.raises(TeamError):
        TeamStore(tmp_path).save(name, TEAM)


def test_missing_and_corrupt_teams(tmp_path):
    store = TeamStore(tmp_path)
    with pytest.raises(TeamError, match="No saved team"):
        store.load("ghosts")
    store.dir.mkdir(parents=True)
    (store.dir / "broken.json").write_text("{not json")
    assert store.list() == []  # skipped, not crashed
    with pytest.raises(TeamError, match="unreadable"):
        store.load("broken")


def drain_until(ws, kind):
    while True:
        ev = ws.receive_json()
        if ev["type"] == kind:
            return ev


def test_websocket_team_round_trip(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    store = TeamStore(tmp_path)
    store.save("Mixed", [*TEAM, Agent("Gro", "builder", "", provider="groq")])
    app = create_app(MockBackend(delay=0), searcher=None, teams=store)
    with TestClient(app) as client, client.websocket_connect("/ws/t") as ws:
        hello = ws.receive_json()
        assert len(hello["agents"]) == 4 and hello["whiteboard"] == ""

        # Groq isn't configured here, so Gro is skipped with a reason, Olly loads.
        ws.send_json({"type": "load_team", "name": "Mixed"})
        roster = drain_until(ws, "roster")
        assert [a["name"] for a in roster["agents"]] == ["Olly"]
        assert "skipped Gro" in drain_until(ws, "notice")["text"]

        ws.send_json({"type": "save_team", "name": "Just Olly"})
        assert drain_until(ws, "teams")["saved"] == "Just Olly"
        assert {t["name"] for t in client.get("/api/teams").json()} == {"Mixed", "Just Olly"}

        ws.send_json({"type": "load_team", "name": "../secrets"})
        assert "Team names" in drain_until(ws, "agent_error")["text"]

        ws.send_json({"type": "delete_team", "name": "Mixed"})
        assert [t["name"] for t in drain_until(ws, "teams")["teams"]] == ["Just Olly"]

        ws.send_json({"type": "set_whiteboard", "text": "# Human notes"})
        board = drain_until(ws, "whiteboard")
        assert board == {"type": "whiteboard", "text": "# Human notes", "by": "Human"}


def test_config_endpoint(tmp_path):
    with TestClient(create_app(MockBackend(delay=0), searcher=None, teams=TeamStore(tmp_path))) as client:
        cfg = client.get("/api/config").json()
        assert cfg["search"] is None and cfg["mock"] is True
