"""FastAPI app: serves the UI, provider info, and a WebSocket per room."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import httpx

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from .agents import DEFAULT_ROSTER, EXTRA_ROSTER, Agent, AgentError
from .llm import Backend, MockBackend, RoutingBackend
from .providers import PROVIDERS, auth_headers, list_models
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

# When a provider's default model has disappeared from its live list (free catalogues churn),
# pick a replacement whose id contains one of these hints, skipping non-chat models.
MODEL_HINTS = {
    "gemini": ("flash", "pro"),
    "groq": ("llama-3.3", "llama", "qwen", "gemma"),
    "openrouter": (":free",),
    "mistral": ("small", "medium", "large"),
    "huggingface": ("instruct", "chat"),
    "custom": ("",),
    "ollama": ("",),
}
NOT_CHAT = ("embed", "tts", "audio", "whisper", "image", "vision-only", "guard", "moderation", "live", "transcribe")


@dataclass
class ProviderCheck:
    id: str
    ok: bool
    model: str | None
    note: str


def pick_model(pid: str, default: str, ids: list[str]) -> str:
    if not ids or default in ids:
        return default
    chat = [i for i in ids if not any(bad in i.lower() for bad in NOT_CHAT)] or ids
    for hint in MODEL_HINTS.get(pid, ("",)):
        matches = [i for i in chat if hint in i.lower()]
        if matches:
            return matches[0]
    return chat[0]


def check_provider(pid: str) -> ProviderCheck:
    """Is this provider usable right now? Lists its models, which proves the key works
    (or that Ollama is running) without spending any tokens."""
    spec = PROVIDERS[pid]
    if pid == "anthropic":
        if not spec.configured:
            return ProviderCheck(pid, False, None, 'not installed (pip install -e ".[claude]")')
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            return ProviderCheck(pid, False, None, "no key")
        return ProviderCheck(pid, True, spec.default_model, "key set (not verified)")
    if not spec.configured:
        return ProviderCheck(pid, False, None, "no key" if pid != "custom" else "no CUSTOM_LLM_BASE_URL")
    try:
        r = httpx.get(f"{spec.url}/models", headers=auth_headers(spec), timeout=8.0)
    except httpx.HTTPError:
        return ProviderCheck(pid, False, None, "not running" if pid == "ollama" else "unreachable")
    if r.status_code in (401, 403) or (r.status_code == 400 and "key" in r.text.lower()):  # Gemini: 400
        return ProviderCheck(pid, False, None, f"key rejected (HTTP {r.status_code}): check {spec.key_env} in keys.env")
    if r.status_code >= 400:
        return ProviderCheck(pid, False, None, f"HTTP {r.status_code}")
    try:
        ids = sorted(m["id"].removeprefix("models/") for m in r.json().get("data", []) if isinstance(m, dict) and "id" in m)
    except ValueError:
        ids = []
    if pid == "ollama" and not ids:
        return ProviderCheck(pid, False, None, "running, but no models pulled (ollama pull llama3.2:1b)")
    model = pick_model(pid, spec.default_model, ids)
    note = "ok" if model == spec.default_model else f"ok (default model unavailable, using {model})"
    return ProviderCheck(pid, True, model, note)


_checks: list[ProviderCheck] | None = None


def check_providers(refresh: bool = False) -> list[ProviderCheck]:
    global _checks
    if _checks is None or refresh:
        with ThreadPoolExecutor(max_workers=len(AUTO_ORDER)) as pool:
            _checks = list(pool.map(check_provider, AUTO_ORDER))
    return _checks


def usable_providers() -> list[str]:
    return [c.id for c in check_providers() if c.ok]


def default_roster() -> list[Agent]:
    """The starter team. AGENT_COLLAB_DEFAULT_PROVIDER / _MODEL move it off Claude, e.g. to
    run the whole room for free on Ollama or Groq. `auto` builds a mixed team: every usable
    provider gets at least one agent (up to 8), using a model that provider actually lists."""
    provider = os.environ.get("AGENT_COLLAB_DEFAULT_PROVIDER", "anthropic")
    model = os.environ.get("AGENT_COLLAB_DEFAULT_MODEL") or None
    if provider == "auto":
        usable = [c for c in check_providers() if c.ok]
        if not usable:
            return list(DEFAULT_ROSTER)
        candidates = [*DEFAULT_ROSTER, *EXTRA_ROSTER]
        size = max(len(DEFAULT_ROSTER), min(len(usable), len(candidates)))
        return [
            dataclasses.replace(candidates[i], provider=usable[i % len(usable)].id, model=usable[i % len(usable)].model)
            for i in range(size)
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
        print("Checking providers…", flush=True)
        lines = ["Provider check:"]
        for c in check_providers():
            unset = c.note.startswith(("no ", "not running", "not installed"))
            mark = "OK " if c.ok else ("-- " if unset else "!! ")
            lines.append(f"  {mark}{PROVIDERS[c.id].label:<28} {c.model + '  ' if c.ok else ''}{c.note}")
        team = default_roster()
        if any(c.ok for c in check_providers()):
            lines.append("Starter team:")
            lines += [f"  {a.name:<4} {a.role:<13} {PROVIDERS[a.provider].label} · {a.model_id}" for a in team]
        else:
            lines.append("No provider is usable yet: add a key to keys.env (see keys.env.example), then rerun.")
        print("\n".join(lines), flush=True)
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
