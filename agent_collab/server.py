"""FastAPI app: serves the UI, provider info, and a WebSocket per room."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import os
from pathlib import Path

import httpx

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from .agents import DEFAULT_ROSTER, Agent, AgentError
from .llm import Backend, MockBackend, RoutingBackend
from .providers import PROVIDERS, list_models
from .room import HUMAN, Room
from .search import Searcher, make_searcher
from .teams import TeamError, TeamStore

STATIC = Path(__file__).parent / "static"


def make_backend() -> Backend:
    if os.environ.get("AGENT_COLLAB_MOCK") == "1":
        return MockBackend(delay=0.04)
    return RoutingBackend()


# Preference order for AGENT_COLLAB_DEFAULT_PROVIDER=auto: capable free tiers first, local last.
AUTO_ORDER = ("gemini", "groq", "openrouter", "mistral", "huggingface", "anthropic", "custom", "ollama")


def _ollama_running() -> bool:
    try:
        return httpx.get(f"{PROVIDERS['ollama'].url}/models", timeout=1.0).status_code == 200
    except httpx.HTTPError:
        return False


def usable_providers() -> list[str]:
    """Providers that can actually answer right now: a key is set (or, for Ollama, the
    server is up). Claude also needs its SDK installed and a key in the environment."""
    usable = []
    for pid in AUTO_ORDER:
        spec = PROVIDERS[pid]
        if pid == "ollama":
            ok = _ollama_running()
        elif pid == "anthropic":
            ok = spec.configured and bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
        else:
            ok = spec.configured
        if ok:
            usable.append(pid)
    return usable


def default_roster() -> list[Agent]:
    """The starter team. AGENT_COLLAB_DEFAULT_PROVIDER / _MODEL move it off Claude, e.g. to
    run the whole room for free on Ollama or Groq; `auto` spreads the four starter agents
    across every provider you have set up, so a mixed-model team works out of the box."""
    provider = os.environ.get("AGENT_COLLAB_DEFAULT_PROVIDER", "anthropic")
    model = os.environ.get("AGENT_COLLAB_DEFAULT_MODEL") or None
    if provider == "auto":
        usable = usable_providers() or ["anthropic"]
        return [
            dataclasses.replace(a, provider=usable[i % len(usable)], model=None)
            for i, a in enumerate(DEFAULT_ROSTER)
        ]
    if provider not in PROVIDERS:
        raise SystemExit(f"AGENT_COLLAB_DEFAULT_PROVIDER={provider!r} is not 'auto' or one of {sorted(PROVIDERS)}")
    return [dataclasses.replace(a, provider=provider, model=model) for a in DEFAULT_ROSTER]


_UNSET = object()


def create_app(
    backend: Backend | None = None, searcher: Searcher | None | object = _UNSET, teams: TeamStore | None = None
) -> FastAPI:
    app = FastAPI(title="agent-collab")
    rooms: dict[str, Room] = {}
    shared_backend = backend or make_backend()
    shared_searcher = make_searcher() if searcher is _UNSET else searcher
    team_store = teams or TeamStore()

    def get_room(room_id: str) -> Room:
        if room_id not in rooms:
            rooms[room_id] = Room(room_id, default_roster(), shared_backend, searcher=shared_searcher)
        return rooms[room_id]

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"ok": True, "rooms": len(rooms)}

    @app.get("/api/providers")
    async def providers() -> list[dict]:
        return [p.to_dict() for p in PROVIDERS.values()]

    @app.get("/api/config")
    async def config() -> dict:
        return {
            "search": shared_searcher.name if shared_searcher else None,
            "mock": isinstance(shared_backend, MockBackend),
            "teams_dir": str(team_store.dir),
        }

    @app.get("/api/teams")
    async def list_teams() -> list[dict]:
        return team_store.list()

    @app.get("/api/providers/{provider_id}/models")
    async def provider_models(provider_id: str) -> dict:
        spec = PROVIDERS.get(provider_id)
        if spec is None:
            raise HTTPException(404, "unknown provider")
        if not spec.configured or isinstance(shared_backend, MockBackend):
            return {"models": [spec.default_model]}
        return {"models": await list_models(spec) or [spec.default_model]}

    @app.websocket("/ws/{room_id}")
    async def room_socket(ws: WebSocket, room_id: str) -> None:
        await ws.accept()
        room = get_room(room_id)
        queue = room.subscribe()
        await ws.send_json(room.snapshot())

        async def pump() -> None:
            while True:
                await ws.send_json(await queue.get())

        pump_task = asyncio.create_task(pump())
        try:
            while True:
                data = await ws.receive_json()
                kind = data.get("type")
                try:
                    if kind == "say" and str(data.get("text", "")).strip():
                        await room.post_human(str(data["text"]).strip())
                    elif kind == "stop":
                        await room.stop()
                    elif kind == "add_agent" and isinstance(data.get("agent"), dict):
                        room.add_agent(data["agent"])
                    elif kind == "remove_agent":
                        room.remove_agent(str(data.get("name", "")))
                    elif kind == "set_whiteboard":
                        room.set_whiteboard(str(data.get("text", "")), by=HUMAN)
                    elif kind == "save_team":
                        name = str(data.get("name", ""))
                        team_store.save(name, room.agents)
                        queue.put_nowait({"type": "teams", "teams": team_store.list(), "saved": name.strip()})
                    elif kind == "load_team":
                        name = str(data.get("name", ""))
                        room.load_roster(team_store.load(name), label=name.strip())
                    elif kind == "delete_team":
                        team_store.delete(str(data.get("name", "")))
                        queue.put_nowait({"type": "teams", "teams": team_store.list()})
                except (AgentError, TeamError) as e:
                    # Only the sender needs to hear about their own invalid input.
                    queue.put_nowait({"type": "agent_error", "text": str(e)})
        except WebSocketDisconnect:
            pass
        finally:
            room.unsubscribe(queue)
            pump_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pump_task

    return app


app = create_app()


def main() -> None:
    import uvicorn

    mock = os.environ.get("AGENT_COLLAB_MOCK") == "1"
    default = os.environ.get("AGENT_COLLAB_DEFAULT_PROVIDER", "anthropic")
    if not mock and default == "auto":
        usable = usable_providers()
        if usable:
            team = ", ".join(f"{a.name}: {PROVIDERS[a.provider].label}" for a in default_roster())
            print(f"Providers ready: {', '.join(PROVIDERS[p].label for p in usable)}\nStarter team: {team}", flush=True)
        else:
            print("warning: no provider keys found. Add one to keys.env (see keys.env.example).", flush=True)
    elif not mock and default in PROVIDERS and not PROVIDERS[default].configured:
        hint = (
            'pip install -e ".[claude]"' if default == "anthropic"
            else f"set {PROVIDERS[default].key_env or PROVIDERS[default].base_url_env}"
        )
        print(
            f"warning: the starter team uses {PROVIDERS[default].label}, which isn't set up ({hint}).\n"
            "         Try AGENT_COLLAB_MOCK=1 for a demo, or AGENT_COLLAB_DEFAULT_PROVIDER=ollama for free local models.",
            flush=True,
        )

    uvicorn.run(
        "agent_collab.server:app",
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
    )
