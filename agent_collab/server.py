"""FastAPI app: serves the UI and a WebSocket per room."""

from __future__ import annotations

import asyncio
import contextlib
import os
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from .agents import DEFAULT_ROSTER
from .llm import Backend, ClaudeBackend, MockBackend
from .room import Room

STATIC = Path(__file__).parent / "static"


def make_backend() -> Backend:
    if os.environ.get("AGENT_COLLAB_MOCK") == "1":
        return MockBackend(delay=0.04)
    return ClaudeBackend()


def create_app(backend: Backend | None = None) -> FastAPI:
    app = FastAPI(title="agent-collab")
    rooms: dict[str, Room] = {}
    shared_backend = backend or make_backend()

    def get_room(room_id: str) -> Room:
        if room_id not in rooms:
            rooms[room_id] = Room(room_id, list(DEFAULT_ROSTER), shared_backend)
        return rooms[room_id]

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"ok": True, "rooms": len(rooms)}

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
                if kind == "say" and str(data.get("text", "")).strip():
                    await room.post_human(str(data["text"]).strip())
                elif kind == "stop":
                    await room.stop()
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
