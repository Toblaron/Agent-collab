"""Load keys.env forgivingly: people paste keys from phones and web pages.

Handles `KEY= value`, quotes, `export KEY=...`, Windows line endings and
invisible characters, and reports anything that still looks wrong.
"""

from __future__ import annotations

import difflib
import os
import re
from pathlib import Path

KNOWN = (
    "GEMINI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY", "MISTRAL_API_KEY", "HF_TOKEN",
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "TAVILY_API_KEY", "BRAVE_API_KEY", "SEARXNG_URL",
    "CUSTOM_LLM_BASE_URL", "CUSTOM_LLM_API_KEY", "CUSTOM_LLM_MODEL", "OLLAMA_BASE_URL",
    "AGENT_COLLAB_DEFAULT_PROVIDER", "AGENT_COLLAB_DEFAULT_MODEL", "AGENT_COLLAB_SEARCH",
)
# Expected prefixes, used only to warn ("this looks like a Groq key on the Gemini line").
PREFIXES = {"GROQ_API_KEY": ("gsk_",), "OPENROUTER_API_KEY": ("sk-or-",), "HF_TOKEN": ("hf_",),
            "ANTHROPIC_API_KEY": ("sk-ant-",), "GEMINI_API_KEY": ("AIza", "AQ."), "TAVILY_API_KEY": ("tvly-",),
            "OPENAI_API_KEY": ("sk-",)}  # last: sk-ant-/sk-or- above must win the "looks like a … key" lookup
INVISIBLE = re.compile(r"[​-‏  ﻿ \r]")


def clean(value: str) -> str:
    value = INVISIBLE.sub("", value).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'“”‘’":
        value = value[1:-1]
    return value.strip("\"'“”‘’ ").strip()


def load_keys(path: Path) -> tuple[dict[str, str], list[str]]:
    """Read keys.env into os.environ (without overriding variables already set).
    Returns (keys found, human-readable problems)."""
    found: dict[str, str] = {}
    problems: list[str] = []
    if not path.exists():
        return found, problems
    for n, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        line = INVISIBLE.sub("", raw).strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            problems.append(f"line {n}: no '=' sign; a key must look like NAME=value")
            continue
        name, value = line.split("=", 1)
        name, value = name.strip().upper(), clean(value)
        if name not in KNOWN:
            guess = difflib.get_close_matches(name, KNOWN, n=1)
            problems.append(f"line {n}: unknown name {name!r}" + (f" (did you mean {guess[0]}?)" if guess else ""))
            continue
        if not value:
            continue
        if " " in value:
            problems.append(f"line {n}: {name} has a space in the middle; keys never contain spaces")
        prefixes = PREFIXES.get(name)
        if prefixes and not value.startswith(prefixes):
            owner = next((k for k, p in PREFIXES.items() if value.startswith(p)), None)
            hint = f"; it looks like a {owner} key" if owner else ""
            problems.append(f"line {n}: {name} usually starts with {' or '.join(map(repr, prefixes))}{hint}")
        found[name] = value
        if not os.environ.get(name):
            os.environ[name] = value
    return found, problems


def mask(value: str) -> str:
    return f"{value[:4]}…{value[-4:]} ({len(value)} chars)" if len(value) > 10 else f"{'*' * len(value)} ({len(value)} chars)"
