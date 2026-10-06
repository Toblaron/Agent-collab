"""FastAPI app: serves the UI, provider info, and a WebSocket per room."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from .agents import DEFAULT_ROSTER, Agent, AgentError
from .llm import Backend, MockBackend, RoutingBackend
from .providers import PROVIDERS, list_models
from .room import Room

STATIC = Path(__file__).parent / "static"


def make_backend() -> Backend:
    if os.environ.get("AGENT_COLLAB_MOCK") == "1":
        return MockBackend(delay=0.04)
    return RoutingBackend()


def default_roster() -> list[Agent]:
    """The starter team. AGENT_COLLAB_DEFAULT_PROVIDER / _MODEL move it off Claude,
    e.g. to run the whole room for free on Ollama or Groq."""
    provider = os.environ.get("AGENT_COLLAB_DEFAULT_PROVIDER", "anthropic")
    if provider not in PROVIDERS:
        raise SystemExit(f"AGENT_COLLAB_DEFAULT_PROVIDER={provider!r} is not one of {sorted(PROVIDERS)}")
    model = os.environ.get("AGENT_COLLAB_DEFAULT_MODEL") or None
    return [dataclasses.replace(a, provider=provider, model=model) for a in DEFAULT_ROSTER]


def create_app(backend: Backend | None = None) -> FastAPI:
    app = FastAPI(title="agent-collab")
    rooms: dict[str, Room] = {}
    shared_backend = backend or make_backend()

    def get_room(room_id: str) -> Room:
        if room_id not in rooms:
            rooms[room_id] = Room(room_id, default_roster(), shared_backend)
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
                except AgentError as e:
                    # Only the sender needs to hear about their own invalid form input.
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

    uvicorn.run(
        "agent_collab.server:app",
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
    )
