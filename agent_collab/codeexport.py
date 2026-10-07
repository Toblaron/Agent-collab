"""Pull the code the team wrote out of a room, as real files in a zip.

Agents can't write files themselves (they only chat), so this is how their work leaves the
room: every fenced code block becomes a file. A file name comes from the block's first-line
comment (`// Entity.cs`, `# file: app.py`) or from a name mentioned just above the block
(`**Player.cs**`); a later version of the same file replaces an earlier one.
"""

from __future__ import annotations

import io
import re
import zipfile
from datetime import datetime

FENCE_RE = re.compile(r"```[ \t]*([\w+#.-]*)[^\n]*\n(.*?)(?:\n```|\Z)", re.DOTALL)
NAME = r"([\w./-]*[\w-]\.(?:[A-Za-z][\w]{0,7}))"
COMMENT_NAME_RE = re.compile(r"^\s*(?://|#|--|;|/\*|<!--)\s*(?:file(?:name)?\s*[:=]\s*)?(?:res://)?" + NAME + r"\b")
MENTION_NAME_RE = re.compile(r"[`*]{1,3}(?:res://)?" + NAME + r"[`*]{1,3}")
EXT = {"csharp": "cs", "cs": "cs", "c#": "cs", "python": "py", "py": "py", "javascript": "js", "js": "js",
       "typescript": "ts", "ts": "ts", "html": "html", "css": "css", "json": "json", "yaml": "yml", "yml": "yml",
       "bash": "sh", "sh": "sh", "shell": "sh", "gdscript": "gd", "gd": "gd", "rust": "rs", "go": "go",
       "java": "java", "cpp": "cpp", "c": "c", "sql": "sql", "toml": "toml", "ini": "ini", "xml": "xml",
       "markdown": "md", "md": "md", "lua": "lua", "kotlin": "kt", "swift": "swift", "tscn": "tscn"}
CODE_EXTS = set(EXT.values()) | {"tres", "godot", "csproj", "sln", "cfg", "txt", "h", "hpp", "jsx", "tsx", "vue"}


def _safe(name: str) -> str | None:
    parts = [p for p in name.replace("\\", "/").split("/") if p not in ("", ".")]
    if not parts or ".." in parts:
        return None
    return "/".join(parts)[:120]


def _name_for(code: str, before: str) -> str | None:
    first = code.lstrip("\n").split("\n", 1)[0]
    m = COMMENT_NAME_RE.match(first)
    if m and m.group(1).rsplit(".", 1)[-1].lower() in CODE_EXTS:
        return _safe(m.group(1))
    near = "\n".join(before.rstrip().split("\n")[-2:])
    hits = [h for h in MENTION_NAME_RE.findall(near) if h.rsplit(".", 1)[-1].lower() in CODE_EXTS]
    return _safe(hits[-1]) if hits else None


def collect_files(state: dict) -> tuple[dict[str, str], dict[str, str]]:
    """({path: content}, {path: who wrote the kept version})."""
    files: dict[str, str] = {}
    authors: dict[str, str] = {}
    snippet = 0
    for m in state.get("messages") or []:
        if m.get("author") in ("Human", "Search"):
            continue
        text = m.get("text", "")
        for block in FENCE_RE.finditer(text):
            lang, code = block.group(1).lower(), block.group(2)
            if lang == "whiteboard" or not code.strip():
                continue
            name = _name_for(code, text[: block.start()])
            if name is None:
                snippet += 1
                name = f"snippets/{snippet:03d}-{m.get('author', 'agent').lower()}.{EXT.get(lang, 'txt')}"
            files[name] = code.rstrip() + "\n"
            authors[name] = m.get("author", "?")
    return files, authors


def code_zip(state: dict) -> bytes | None:
    files, authors = collect_files(state)
    if not files:
        return None
    title = state.get("title") or state.get("id")
    lines = [f"# {title}", "", f"Code from the agent-collab room `{state.get('id')}`, exported "
             f"{datetime.now():%Y-%m-%d %H:%M}. Agents write code in chat only; nothing here has been "
             "compiled or run yet.", "", "| File | Last written by |", "|---|---|"]
    lines += [f"| `{p}` | {authors[p]} |" for p in sorted(files)]
    if state.get("whiteboard"):
        lines += ["", "## Whiteboard", "", state["whiteboard"].strip()]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("README.md", "\n".join(lines) + "\n")
        for path, content in sorted(files.items()):
            z.writestr(path if path != "README.md" else "code/README.md", content)
    return buf.getvalue()
