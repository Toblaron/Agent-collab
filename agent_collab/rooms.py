"""Saved rooms: each room's conversation, whiteboard, team and settings as a JSON file,
so closing the server (or Termux) never loses a chat.

Location: $AGENT_COLLAB_DATA_DIR/rooms/<room-id>.json (default ~/.agent-collab/rooms/).
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

from .teams import default_data_dir

ROOM_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


def valid_room_id(room_id: str) -> bool:
    return bool(ROOM_ID_RE.fullmatch(room_id))


class RoomStore:
    def __init__(self, root: Path | None = None):
        self.dir = (root or default_data_dir()) / "rooms"

    def _path(self, room_id: str) -> Path:
        if not valid_room_id(room_id):
            raise ValueError(f"invalid room id {room_id!r}")
        return self.dir / f"{room_id}.json"

    def save(self, state: dict) -> None:
        path = self._path(state["id"])
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False))
        tmp.replace(path)  # atomic: a crash mid-write never leaves a truncated room

    def load(self, room_id: str) -> dict | None:
        path = self._path(room_id)
        try:
            data = json.loads(path.read_text())
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            # Keep the unreadable file for inspection instead of silently overwriting it.
            path.replace(path.with_suffix(f".corrupt-{int(time.time())}.json"))
            return None
        return data if isinstance(data, dict) else None

    def delete(self, room_id: str) -> bool:
        path = self._path(room_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def list(self) -> list[dict]:
        if not self.dir.exists():
            return []
        rooms = []
        for path in self.dir.glob("*.json"):
            if not valid_room_id(path.stem):
                continue  # skips .corrupt-*.json backups and anything hand-placed
            try:
                data = json.loads(path.read_text())
                messages = data.get("messages") or []
                rooms.append(
                    {
                        "id": data.get("id", path.stem),
                        "title": data.get("title") or path.stem,
                        "updated_at": data.get("updated_at", path.stat().st_mtime),
                        "messages": len(messages),
                        "agents": [a.get("name") for a in data.get("agents") or [] if isinstance(a, dict)],
                    }
                )
            except (OSError, ValueError, AttributeError):
                continue
        return sorted(rooms, key=lambda r: r["updated_at"], reverse=True)


def export_markdown(state: dict) -> str:
    """A readable transcript: who said what (with model), plus the final whiteboard."""
    def when(ts: float) -> str:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")

    lines = [f"# {state.get('title') or state.get('id')}", ""]
    agents = state.get("agents") or []
    if agents:
        lines.append("**Team:** " + ", ".join(f"{a['name']} ({a.get('role', '')}, {a.get('model', '?')})" for a in agents))
        lines.append("")
    if state.get("whiteboard"):
        by = f" (last edit: {state['whiteboard_by']})" if state.get("whiteboard_by") else ""
        lines += [f"## Whiteboard{by}", "", state["whiteboard"].strip(), ""]
    lines += ["## Conversation", ""]
    for m in state.get("messages") or []:
        meta = m.get("meta") or {}
        extra = f" · {meta['model']}" if meta.get("model") else ""
        extra += f" · {meta['secs']}s" if meta.get("secs") is not None else ""
        lines += [f"**{m['author']}** · {when(m.get('ts', 0))}{extra}", "", m.get("text", "").strip(), ""]
    return "\n".join(lines).rstrip() + "\n"
