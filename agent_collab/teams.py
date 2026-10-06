"""Saved teams: named rosters stored as JSON files.

Location: $AGENT_COLLAB_DATA_DIR/teams/ (default ~/.agent-collab/teams/).
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from .agents import Agent

TEAM_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _-]{0,39}$")
MAX_TEAMS = 200


class TeamError(ValueError):
    pass


def default_data_dir() -> Path:
    return Path(os.environ.get("AGENT_COLLAB_DATA_DIR") or Path.home() / ".agent-collab")


class TeamStore:
    def __init__(self, root: Path | None = None):
        self.dir = (root or default_data_dir()) / "teams"

    def _path(self, name: str) -> Path:
        name = name.strip()
        if not TEAM_NAME_RE.match(name):
            raise TeamError("Team names are 1-40 letters, digits, spaces, - or _, starting with a letter or digit.")
        # The regex already rules out path separators and dots; the slug just makes a tidy filename.
        return self.dir / (re.sub(r"[ _]+", "-", name.lower()) + ".json")

    def list(self) -> list[dict]:
        if not self.dir.exists():
            return []
        teams = []
        for path in sorted(self.dir.glob("*.json")):
            try:
                data = json.loads(path.read_text())
                teams.append(
                    {
                        "name": data["name"],
                        "agents": [a["name"] for a in data["agents"]],
                        "saved_at": data.get("saved_at", 0),
                    }
                )
            except (OSError, ValueError, KeyError, TypeError):
                continue  # skip a hand-edited or half-written file rather than break the list
        return sorted(teams, key=lambda t: t["saved_at"], reverse=True)

    def save(self, name: str, agents: list[Agent]) -> None:
        path = self._path(name)
        if not agents:
            raise TeamError("There's nobody in the room to save.")
        if not path.exists() and len(self.list()) >= MAX_TEAMS:
            raise TeamError(f"You already have {MAX_TEAMS} saved teams; delete some first.")
        self.dir.mkdir(parents=True, exist_ok=True)
        data = {"name": name.strip(), "saved_at": time.time(), "agents": [a.to_dict() for a in agents]}
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(path)  # atomic: a crash never leaves a truncated team file

    def load(self, name: str) -> list[dict]:
        path = self._path(name)
        try:
            data = json.loads(path.read_text())
        except FileNotFoundError:
            raise TeamError(f"No saved team called {name!r}.") from None
        except (OSError, ValueError) as e:
            raise TeamError(f"Team file for {name!r} is unreadable: {e}") from None
        agents = data.get("agents")
        if not isinstance(agents, list):
            raise TeamError(f"Team file for {name!r} has no agent list.")
        return [a for a in agents if isinstance(a, dict)]

    def delete(self, name: str) -> None:
        path = self._path(name)
        if not path.exists():
            raise TeamError(f"No saved team called {name!r}.")
        path.unlink()
