"""Tool use via plain-text markers, so every model can use tools, from Claude down
to a 3B local model with no function-calling support:

    ```whiteboard
    <the full new whiteboard document>
    ```

    [[search: some query]]

After an agent finishes speaking, the room pulls these out of its message,
applies them, and leaves a short marker in the visible transcript.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

WHITEBOARD = "whiteboard"
SEARCH = "search"
TOOLS = (WHITEBOARD, SEARCH)

MAX_WHITEBOARD_CHARS = 20_000
MAX_SEARCHES_PER_MESSAGE = 2

# Unterminated fence (model hit max_tokens mid-block) still counts: take everything to the end.
WHITEBOARD_RE = re.compile(r"```[ \t]*whiteboard[^\n]*\n(.*?)(?:\n?```|\Z)", re.DOTALL | re.IGNORECASE)
SEARCH_RE = re.compile(r"\[\[\s*search\s*:\s*(.+?)\s*\]\]", re.IGNORECASE)


@dataclass
class Actions:
    whiteboard: str | None = None
    searches: list[str] = field(default_factory=list)


def extract_actions(text: str, tools: tuple[str, ...]) -> tuple[str, Actions]:
    """Return (text for the transcript, actions to apply). Markers for tools the
    agent doesn't have are left in place, untouched."""
    actions = Actions()

    if WHITEBOARD in tools:
        blocks = WHITEBOARD_RE.findall(text)
        if blocks:
            actions.whiteboard = blocks[-1].strip()[:MAX_WHITEBOARD_CHARS]
            text = WHITEBOARD_RE.sub("[updated the whiteboard]", text)

    if SEARCH in tools:
        def take(match: re.Match) -> str:
            query = match.group(1)[:200]
            if len(actions.searches) < MAX_SEARCHES_PER_MESSAGE and query not in actions.searches:
                actions.searches.append(query)
            return f"[searching: {query}]"

        text = SEARCH_RE.sub(take, text)

    return text.strip(), actions


def tool_instructions(tools: tuple[str, ...]) -> str:
    if not tools:
        return ""
    lines = ["Tools you can use (write them exactly like this inside your message):"]
    if WHITEBOARD in tools:
        lines.append(
            "- Shared whiteboard: the team's one shared document, shown to everyone in <whiteboard>. "
            "Use it for plans, decisions, task lists and drafts. To change it, include a block that "
            "starts with ```whiteboard on its own line and ends with ```; it REPLACES the whole "
            "whiteboard, so write the complete updated document, not just your additions."
        )
    if SEARCH in tools:
        lines.append(
            "- Web search: write [[search: your query]] on its own line (at most 2 per message). "
            "Results are posted to the room after your message, and you can respond to them then. "
            "Search when facts, numbers or prior art would change the team's decision; don't search for "
            "things you already know."
        )
    return "\n".join(lines)
