"""FastAPI app: serves the UI, provider info, and a WebSocket per room."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass
from pathlib import Path

import httpx

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, PlainTextResponse

from .agents import DEFAULT_ROSTER, EXTRA_ROSTER, Agent, AgentError
from .llm import Backend, MockBackend, RoutingBackend
from .providers import PROVIDERS, auth_headers, list_models
from .room import HUMAN, Room
from .rooms import RoomStore, export_markdown, valid_room_id
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
    "gemini": ("flash-latest", "flash-lite-latest", "flash", "pro"),
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


def candidate_models(pid: str, default: str, ids: list[str]) -> list[str]:
    """Models worth trying, best first: the default, then listed chat models matching the hints."""
    chat = [i for i in ids if not any(bad in i.lower() for bad in NOT_CHAT)]
    out = [default] if (not ids or default in ids) else []
    for hint in MODEL_HINTS.get(pid, ("",)):
        out += [i for i in chat if hint in i.lower() and i not in out]
    return out or chat[:1] or [default]


def probe_chat(spec, model: str, timeout: float = 12.0) -> tuple[int, str]:
    """One tiny real request: the only way to know a model will actually answer
    (listed models can be retired for new accounts, or overloaded right now)."""
    try:
        r = httpx.post(
            f"{spec.url}/chat/completions",
            headers=auth_headers(spec),
            json={"model": model, "messages": [{"role": "user", "content": "Reply with OK."}], "max_tokens": 5},
            timeout=timeout,
        )
    except httpx.HTTPError:
        return 0, "unreachable"
    return r.status_code, r.text


MAX_PROBES = 3
CHECK_DEADLINE = 20.0  # seconds for the whole startup check; slow providers are assumed fine


def check_provider(pid: str, budget: float = CHECK_DEADLINE) -> ProviderCheck:
    """Is this provider usable right now? Lists its models (proves the key), then sends a
    5-token request to the best candidate, falling back through listed models when one is
    retired or overloaded. Costs a handful of tokens on free tiers."""
    spec = PROVIDERS[pid]
    if pid == "anthropic":
        if not spec.configured:
            return ProviderCheck(pid, False, None, 'not installed (pip install -e ".[claude]")')
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            return ProviderCheck(pid, False, None, "no key")
        return ProviderCheck(pid, True, spec.default_model, "key set (not verified)")
    if not spec.configured:
        return ProviderCheck(pid, False, None, "no key" if pid != "custom" else "no CUSTOM_LLM_BASE_URL")
    stop_at = time.monotonic() + budget
    try:
        r = httpx.get(f"{spec.url}/models", headers=auth_headers(spec), timeout=min(6.0, budget))
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
    if pid == "ollama":  # local: listing is proof enough, and a cold model load can take a minute
        if not ids:
            return ProviderCheck(pid, False, None, "running, but no models pulled (ollama pull llama3.2:1b)")
        model = pick_model(pid, spec.default_model, ids)
        return ProviderCheck(pid, True, model, "ok")

    busy: str | None = None
    tried = []
    for model in candidate_models(pid, spec.default_model, ids)[:MAX_PROBES]:
        remaining = stop_at - time.monotonic()
        if remaining < 2:
            break
        status, body = probe_chat(spec, model, timeout=min(12.0, remaining))
        if status == 200:
            note = "ok" if model == spec.default_model else f"ok (using {model}; {', '.join(tried)} unavailable)"
            return ProviderCheck(pid, True, model, note)
        if status in (401, 403):
            return ProviderCheck(pid, False, None, f"key rejected (HTTP {status}): check {spec.key_env} in keys.env")
        if status in (429, 503) and busy is None:
            busy = model  # works, just busy/limited right now; keep looking for one that answers
        tried.append(model)
    if busy:
        return ProviderCheck(pid, True, busy, "ok, but busy/rate-limited right now (agents will retry)")
    if not tried:  # ran out of time before any test message: the key is valid (listing worked)
        model = candidate_models(pid, spec.default_model, ids)[0]
        return ProviderCheck(pid, True, model, "key ok; slow to answer (not fully tested)")
    return ProviderCheck(pid, False, None, f"no model answered (tried {', '.join(tried)})")


_checks: list[ProviderCheck] | None = None


def _assumed(pid: str) -> ProviderCheck:
    """Result for a provider whose check didn't finish in time: trust the key, keep the default model."""
    spec = PROVIDERS[pid]
    if pid == "ollama":
        has_key = False  # no key to trust; Ollama counts only when it's actually answering
    elif pid == "anthropic":
        has_key = spec.configured and bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    else:
        has_key = spec.configured
    note = "check timed out; using default model" if has_key else "check timed out"
    return ProviderCheck(pid, has_key, spec.default_model if has_key else None, note)


def check_providers(
    refresh: bool = False,
    deadline: float = CHECK_DEADLINE,
    on_result: Callable[[ProviderCheck], None] | None = None,
) -> list[ProviderCheck]:
    """Check every provider in parallel, reporting each as it finishes. Never takes much
    longer than `deadline`: anything still running is assumed fine and checked for real
    the first time an agent talks to it."""
    global _checks
    if _checks is not None and not refresh:
        return _checks
    if os.environ.get("AGENT_COLLAB_SKIP_CHECK") == "1":
        _checks = []
        for pid in AUTO_ORDER:
            c = _assumed(pid)
            c.note = "not checked (fast start)" if c.ok else ("no key" if pid != "ollama" else "not checked")
            _checks.append(c)
        return _checks
    pool = ThreadPoolExecutor(max_workers=len(AUTO_ORDER))
    futures = {pool.submit(check_provider, pid, deadline - 2): pid for pid in AUTO_ORDER}
    results: dict[str, ProviderCheck] = {}
    try:
        for fut in as_completed(futures, timeout=deadline):
            pid = futures[fut]
            try:
                results[pid] = fut.result()
            except Exception as e:  # never let one provider's surprise break startup
                results[pid] = ProviderCheck(pid, False, None, f"check failed ({type(e).__name__})")
            if on_result:
                on_result(results[pid])
    except FuturesTimeout:
        for pid in AUTO_ORDER:
            if pid not in results:
                results[pid] = _assumed(pid)
                if on_result:
                    on_result(results[pid])
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    _checks = [results[pid] for pid in AUTO_ORDER]
    return _checks


def format_check(c: ProviderCheck) -> str:
    unset = c.note.startswith(("no ", "not running", "not installed", "not checked"))
    mark = "OK " if c.ok else ("-- " if unset else "!! ")
    return f"  {mark}{PROVIDERS[c.id].label:<28} {c.model + '  ' if c.ok else ''}{c.note}"


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
MAX_HUMAN_MESSAGE = 20_000


def create_app(
    backend: Backend | None = None,
    searcher: Searcher | None | object = _UNSET,
    teams: TeamStore | None = None,
    room_store: RoomStore | None = None,
) -> FastAPI:
    app = FastAPI(title="agent-collab")
    rooms: dict[str, Room] = {}
    shared_backend = backend or make_backend()
    shared_searcher = make_searcher() if searcher is _UNSET else searcher
    team_store = teams or TeamStore()
    store = room_store or RoomStore()

    def save(room: Room) -> None:
        store.save(room.to_state())

    def get_room(room_id: str) -> Room:
        if room_id not in rooms:
            room = Room(room_id, default_roster(), shared_backend, searcher=shared_searcher, on_change=save)
            saved = store.load(room_id)
            if saved:
                room.load_state(saved)
            rooms[room_id] = room
        return rooms[room_id]

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

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
            "rooms_dir": str(store.dir),
        }

    @app.get("/api/teams")
    async def list_teams() -> list[dict]:
        return team_store.list()

    @app.get("/api/rooms")
    async def list_rooms() -> list[dict]:
        listed = {r["id"]: r for r in store.list()}
        for room in rooms.values():  # live state beats what's on disk (e.g. a room that's mid-reply)
            if room.messages:
                listed[room.id] = {
                    "id": room.id, "title": room.title, "updated_at": room.updated_at,
                    "messages": len(room.messages), "agents": [a.name for a in room.agents],
                    "running": room.running,
                }
        return sorted(listed.values(), key=lambda r: r["updated_at"], reverse=True)

    @app.get("/api/rooms/{room_id}/export.md")
    async def export_room(room_id: str) -> PlainTextResponse:
        if not valid_room_id(room_id):
            raise HTTPException(400, "invalid room id")
        state = rooms[room_id].to_state() if room_id in rooms else store.load(room_id)
        if not state:
            raise HTTPException(404, "no such room")
        return PlainTextResponse(
            export_markdown(state),
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="agent-collab-{room_id}.md"'},
        )

    @app.delete("/api/rooms/{room_id}")
    async def delete_room(room_id: str) -> dict:
        if not valid_room_id(room_id):
            raise HTTPException(400, "invalid room id")
        room = rooms.pop(room_id, None)
        if room:
            await room.stop()
            room._emit({"type": "deleted"})
        return {"deleted": store.delete(room_id) or room is not None}

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
        if not valid_room_id(room_id):
            await ws.close(code=1008, reason="invalid room id")
            return
        await ws.accept()
        room = get_room(room_id)
        queue = room.subscribe()
        await ws.send_json(room.snapshot())

        async def pump() -> None:
            while True:
                await ws.send_json(await queue.get())

        def reply_error(text: str) -> None:
            # Only the sender needs to hear about their own invalid input.
            queue.put_nowait({"type": "agent_error", "text": text})

        pump_task = asyncio.create_task(pump())
        try:
            while True:
                try:
                    data = await ws.receive_json()
                except (ValueError, KeyError):
                    reply_error("That message wasn't valid JSON.")
                    continue
                if not isinstance(data, dict):
                    reply_error("Messages must be JSON objects.")
                    continue
                kind = data.get("type")
                try:
                    await handle(room, kind, data, queue)
                except (AgentError, TeamError) as e:
                    reply_error(str(e))
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            room.unsubscribe(queue)
            pump_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await pump_task

    async def handle(room: Room, kind, data: dict, queue: asyncio.Queue) -> None:
        name = str(data.get("name", ""))
        if kind == "say":
            text = str(data.get("text", "")).strip()
            if len(text) > MAX_HUMAN_MESSAGE:
                raise AgentError(f"That message is too long (max {MAX_HUMAN_MESSAGE:,} characters).")
            if text:
                await room.post_human(text)
        elif kind == "stop":
            await room.stop()
        elif kind == "continue":
            await room.continue_conversation()
        elif kind == "clear":
            await room.clear()
        elif kind == "add_agent" and isinstance(data.get("agent"), dict):
            room.add_agent(data["agent"])
        elif kind == "update_agent" and isinstance(data.get("agent"), dict):
            room.update_agent(name, data["agent"])
        elif kind == "remove_agent":
            room.remove_agent(name)
        elif kind == "set_muted":
            room.set_muted(name, bool(data.get("muted", True)))
        elif kind == "set_max_turns":
            room.set_max_turns(data.get("value"))
        elif kind == "set_whiteboard":
            room.set_whiteboard(str(data.get("text", "")), by=HUMAN)
        elif kind == "save_team":
            team_store.save(name, room.agents)
            queue.put_nowait({"type": "teams", "teams": team_store.list(), "saved": name.strip()})
        elif kind == "load_team":
            room.load_roster(team_store.load(name), label=name.strip())
        elif kind == "delete_team":
            team_store.delete(name)
            queue.put_nowait({"type": "teams", "teams": team_store.list()})
        else:
            raise AgentError(f"Unknown request {kind!r}.")

    return app


app = create_app()


def main() -> None:
    import uvicorn

    mock = os.environ.get("AGENT_COLLAB_MOCK") == "1"
    default = os.environ.get("AGENT_COLLAB_DEFAULT_PROVIDER", "anthropic")
    if not mock and default == "auto":
        print(f"Checking providers (up to {CHECK_DEADLINE:.0f}s; `bash run.sh fast` skips this)…", flush=True)
        checks = check_providers(on_result=lambda c: print(format_check(c), flush=True))
        team = default_roster()
        if any(c.ok for c in checks):
            lines = ["Starter team:"]
            lines += [f"  {a.name:<4} {a.role:<13} {PROVIDERS[a.provider].label} · {a.model_id}" for a in team]
        else:
            lines = ["No provider is usable yet: add a key to keys.env (see keys.env.example), then rerun."]
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

    host, port = os.environ.get("HOST", "127.0.0.1"), int(os.environ.get("PORT", "8000"))
    shown = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    print(f"\n>>> Ready: open http://{shown}:{port} in your browser. Stop with CTRL+C.\n", flush=True)
    uvicorn.run("agent_collab.server:app", host=host, port=port, log_level="warning")
