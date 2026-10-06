"""LLM providers an agent can run on.

`anthropic` uses the Anthropic SDK. Every other provider speaks the
OpenAI-compatible chat completions API, so one backend covers them all.
Base URLs come only from this table or from env vars, never from the UI,
so a browser can't point the server at arbitrary hosts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class ProviderSpec:
    id: str
    label: str
    base_url: str
    key_env: str | None  # None = no key needed (local)
    default_model: str
    note: str
    base_url_env: str | None = None

    @property
    def url(self) -> str:
        if self.base_url_env and os.environ.get(self.base_url_env):
            return os.environ[self.base_url_env].rstrip("/")
        return self.base_url.rstrip("/")

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.key_env) if self.key_env else None

    @property
    def configured(self) -> bool:
        if self.id == "anthropic":
            # Installed is the gate: the SDK also resolves `ant auth login` profiles, which env vars can't reveal.
            import importlib.util

            return importlib.util.find_spec("anthropic") is not None
        if self.id == "custom":
            return bool(os.environ.get("CUSTOM_LLM_BASE_URL"))
        return self.key_env is None or bool(self.api_key)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "default_model": self.default_model,
            "note": self.note,
            "configured": self.configured,
            "key_env": self.key_env,
        }


PROVIDERS: dict[str, ProviderSpec] = {
    p.id: p
    for p in [
        ProviderSpec(
            "anthropic", "Claude (Anthropic)", "https://api.anthropic.com", "ANTHROPIC_API_KEY",
            os.environ.get("AGENT_COLLAB_MODEL", "claude-opus-5-5"), "Paid API",
        ),
        ProviderSpec(
            "ollama", "Ollama (local)", "http://localhost:11434/v1", None,
            "llama3.2", "Free, runs on your machine. `ollama pull <model>` first.", base_url_env="OLLAMA_BASE_URL",
        ),
        ProviderSpec(
            "groq", "Groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY",
            "llama-3.3-70b-versatile", "Free tier with rate limits. Key: console.groq.com",
        ),
        ProviderSpec(
            "gemini", "Google Gemini", "https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY",
            "gemini-flash-lite-latest", "Free tier (each model has its own daily quota). Key: aistudio.google.com",
        ),
        ProviderSpec(
            "openrouter", "OpenRouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY",
            "meta-llama/llama-3.3-70b-instruct:free", "Models ending in `:free` cost nothing. Key: openrouter.ai",
        ),
        ProviderSpec(
            "huggingface", "Hugging Face", "https://router.huggingface.co/v1", "HF_TOKEN",
            "meta-llama/Llama-3.1-8B-Instruct", "Free monthly credits. Token: huggingface.co/settings/tokens",
        ),
        ProviderSpec(
            "mistral", "Mistral", "https://api.mistral.ai/v1", "MISTRAL_API_KEY",
            "mistral-small-latest", "Free experiment tier. Key: console.mistral.ai",
        ),
        ProviderSpec(
            "openai", "OpenAI", "https://api.openai.com/v1", "OPENAI_API_KEY",
            "gpt-5-mini",
            "Paid API. Free daily tokens if you opt in to data sharing (platform.openai.com → "
            "Settings → Data controls). For free OpenAI models with no key, use gpt-oss on Groq/OpenRouter.",
        ),
        ProviderSpec(
            "custom", "Custom (OpenAI-compatible)", "", "CUSTOM_LLM_API_KEY",
            os.environ.get("CUSTOM_LLM_MODEL", "default"),
            "Any OpenAI-compatible server (LM Studio, vLLM, llama.cpp…). Set CUSTOM_LLM_BASE_URL.",
            base_url_env="CUSTOM_LLM_BASE_URL",
        ),
    ]
}


def auth_headers(spec: ProviderSpec) -> dict[str, str]:
    headers = {}
    if spec.api_key:
        headers["Authorization"] = f"Bearer {spec.api_key}"
    if spec.id == "openrouter":
        headers["X-Title"] = "agent-collab"
    return headers


async def list_models(spec: ProviderSpec, client: httpx.AsyncClient | None = None) -> list[str]:
    """Live model list from the provider, so the UI never offers stale IDs."""
    if spec.id == "anthropic":
        import anthropic

        try:
            page = await anthropic.AsyncAnthropic().models.list(limit=50)
            return [m.id for m in page.data]
        except Exception:  # no credentials, network, etc. The default is still usable.
            return [spec.default_model]
    if not spec.url:
        return []
    owns = client is None
    client = client or httpx.AsyncClient(timeout=10)
    try:
        r = await client.get(f"{spec.url}/models", headers=auth_headers(spec))
        r.raise_for_status()
        data = r.json().get("data", [])
        ids = sorted(m["id"] for m in data if isinstance(m, dict) and "id" in m)
        if spec.id == "gemini":
            ids = [i.removeprefix("models/") for i in ids]
        return ids
    except (httpx.HTTPError, ValueError, KeyError):
        return []
    finally:
        if owns:
            await client.aclose()


# When a provider's default model has disappeared from its live list (free catalogues churn),
# pick a replacement whose id contains one of these hints, skipping non-chat models.
MODEL_HINTS = {
    "gemini": ("flash-lite-latest", "flash-lite", "flash-latest", "flash", "pro"),
    "groq": ("llama-3.3", "gpt-oss-120b", "llama", "gpt-oss", "qwen", "gemma"),
    "openai": ("gpt-5-mini", "gpt-5.1-mini", "gpt-4.1-mini", "gpt-4o-mini", "mini", "gpt-5", "gpt-4.1"),
    "openrouter": (":free",),
    "mistral": ("small", "medium", "large"),
    "huggingface": ("instruct", "chat"),
    "custom": ("",),
    "ollama": ("",),
}
NOT_CHAT = ("embed", "tts", "audio", "whisper", "image", "vision-only", "guard", "moderation", "live", "transcribe",
            "realtime", "dall-e", "davinci", "babbage", "sora", "search", "computer-use", "codex", "deep-research")


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


def is_reasoning_openai(model: str) -> bool:
    """OpenAI's reasoning models (gpt-5*, o1/o3/o4…) take different parameters than classic chat models."""
    m = model.lower().removeprefix("openai/")
    return m.startswith("gpt-5") or (len(m) > 1 and m[0] == "o" and m[1].isdigit())


def adapt_body(pid: str, body: dict) -> dict:
    """Provider quirks for an OpenAI-style chat body. OpenAI's own API wants `max_completion_tokens`,
    and its reasoning models reject a custom temperature and spend hidden tokens thinking."""
    if pid != "openai":
        return body
    body = dict(body)
    if "max_tokens" in body:
        body["max_completion_tokens"] = body.pop("max_tokens")
    if is_reasoning_openai(body.get("model", "")):
        body.pop("temperature", None)
        body["reasoning_effort"] = "low"
        # Reasoning tokens count against the limit; leave room for the visible answer too.
        body["max_completion_tokens"] = max(body.get("max_completion_tokens", 0), 2048) + 2048
    return body
