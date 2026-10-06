"""LLM backends: how an agent bids for the floor and how it speaks.

- `ClaudeBackend`: Anthropic SDK (structured-output bids, streamed replies).
- `OpenAICompatBackend`: every other provider (Ollama, Groq, Gemini,
  OpenRouter, Hugging Face, Mistral, custom) via /chat/completions.
- `RoutingBackend`: dispatches each agent to the backend for its provider.
- `MockBackend`: deterministic and offline, for tests and `AGENT_COLLAB_MOCK=1`.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
from typing import AsyncIterator, Protocol

import httpx

try:  # optional: `pip install -e ".[claude]"`
    import anthropic
except ImportError:  # pragma: no cover - exercised on installs without the extra
    anthropic = None
from pydantic import BaseModel, Field

from .agents import Agent, Message, render_transcript
from .providers import PROVIDERS, ProviderSpec, adapt_body, auth_headers, candidate_models

BID_EFFORT = os.environ.get("AGENT_COLLAB_BID_EFFORT", "low")
SPEAK_EFFORT = os.environ.get("AGENT_COLLAB_SPEAK_EFFORT", "medium")
FALLBACK_BETA = "server-side-fallback-2026-07-01"
CLAUDE_MISSING = 'Claude support is not installed. Run: pip install -e ".[claude]"'


class Bid(BaseModel):
    """An agent's answer to 'do you want to speak next?'"""

    urgency: float = Field(description="0.0 = nothing to add, 1.0 = must speak right now")
    reason: str = Field(description="One short phrase: what you would say, or why you are staying quiet")


class Backend(Protocol):
    async def bid(self, agent: Agent, roster: list[Agent], transcript: list[Message], whiteboard: str = "") -> Bid: ...

    def speak(
        self, agent: Agent, roster: list[Agent], transcript: list[Message], whiteboard: str = ""
    ) -> AsyncIterator[str]: ...


BID_INSTRUCTIONS = (
    "Above is the conversation so far. Decide whether YOU should speak next.\n"
    "Score urgency from 0.0 to 1.0:\n"
    "- 0.8-1.0: you were directly addressed (@{name}), or you see a mistake or blocker no one has raised\n"
    "- 0.5-0.7: you have a genuinely new contribution that moves the work forward\n"
    "- 0.1-0.4: you could add something minor\n"
    "- 0.0: you would only be agreeing, repeating, or the task is done\n"
    "Silence is a good default. Do not bid high just to participate."
)

JSON_BID_SUFFIX = (
    '\n\nRespond with ONLY a JSON object and nothing else, for example:\n'
    '{"urgency": 0.3, "reason": "could add a minor point about caching"}'
)

SPEAK_INSTRUCTIONS = "Above is the conversation so far. It is your turn. Write only your next message, as {name}, with no name prefix."


def _prompt(transcript: list[Message], instructions: str, whiteboard: str = "") -> str:
    board = whiteboard.strip() or "(empty)"
    return (
        f"<whiteboard>\n{board}\n</whiteboard>\n\n"
        f"<transcript>\n{render_transcript(transcript)}\n</transcript>\n\n{instructions}"
    )


def _failed_bid(reason: str) -> Bid:
    # A failed bid means "stay quiet this round", not a crashed room. The reason shows in the sidebar.
    return Bid(urgency=0.0, reason=reason[:120])


def _clamp(bid: Bid) -> Bid:
    bid.urgency = max(0.0, min(1.0, bid.urgency))
    return bid


# ---- Claude ---------------------------------------------------------------


class ClaudeBackend:
    def __init__(self, client: "anthropic.AsyncAnthropic | None" = None):
        self._client = client

    @property
    def client(self) -> "anthropic.AsyncAnthropic":
        # Lazy so importing the app (tests, mock mode, free-only setups) never needs credentials.
        if self._client is None:
            if anthropic is None:
                raise ProviderError(CLAUDE_MISSING)
            self._client = anthropic.AsyncAnthropic()
        return self._client

    async def bid(self, agent: Agent, roster: list[Agent], transcript: list[Message], whiteboard: str = "") -> Bid:
        if anthropic is None and self._client is None:
            return _failed_bid("(Claude not installed)")
        try:
            response = await self.client.beta.messages.parse(
                model=agent.model_id,
                max_tokens=4000,
                system=agent.system_prompt(roster),
                messages=[{"role": "user", "content": _prompt(transcript, BID_INSTRUCTIONS.format(name=agent.name), whiteboard)}],
                output_format=Bid,
                output_config={"effort": BID_EFFORT},
                cache_control={"type": "ephemeral"},
                betas=[FALLBACK_BETA],
                fallbacks="default",
            )
        except anthropic.AnthropicError as e:
            return _failed_bid(f"(error: {type(e).__name__})")
        if response.stop_reason == "refusal" or response.parsed_output is None:
            return _failed_bid("(declined)")
        return _clamp(response.parsed_output)

    async def speak(
        self, agent: Agent, roster: list[Agent], transcript: list[Message], whiteboard: str = ""
    ) -> AsyncIterator[str]:
        async with self.client.beta.messages.stream(
            model=agent.model_id,
            max_tokens=16000,
            system=agent.system_prompt(roster),
            messages=[{"role": "user", "content": _prompt(transcript, SPEAK_INSTRUCTIONS.format(name=agent.name), whiteboard)}],
            output_config={"effort": SPEAK_EFFORT},
            cache_control={"type": "ephemeral"},
            betas=[FALLBACK_BETA],
            fallbacks="default",
        ) as stream:
            async for text in stream.text_stream:
                yield text
            final = await stream.get_final_message()
            if final.stop_reason == "refusal":
                yield "\n\n_(I can't help with that part.)_"


# ---- OpenAI-compatible (free / open models) -------------------------------


def parse_bid(text: str) -> Bid:
    """Free models don't all support JSON mode, so parse leniently:
    strip reasoning blocks and code fences, take the first {...}, fall back to a bare number."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            return _clamp(Bid(urgency=float(data.get("urgency", 0)), reason=str(data.get("reason", ""))[:200]))
        except (ValueError, TypeError):
            pass
    number = re.search(r"urgency\W{0,4}([01](?:\.\d+)?)", text, flags=re.IGNORECASE)
    if number:
        return _clamp(Bid(urgency=float(number.group(1)), reason="(unstructured bid)"))
    return _failed_bid("(unparseable bid)")


class ThinkFilter:
    """Drops <think>...</think> spans from a token stream (reasoning models like
    DeepSeek-R1 / Qwen3 emit them inline), even when tags are split across chunks."""

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self) -> None:
        self.buf = ""
        self.inside = False

    def feed(self, chunk: str) -> str:
        self.buf += chunk
        out = []
        while True:
            tag = self.CLOSE if self.inside else self.OPEN
            idx = self.buf.find(tag)
            if idx >= 0:
                if not self.inside:
                    out.append(self.buf[:idx])
                self.buf = self.buf[idx + len(tag) :]
                self.inside = not self.inside
                continue
            # Hold back a possible partial tag at the end of the buffer.
            keep = next((k for k in range(len(tag) - 1, 0, -1) if self.buf.endswith(tag[:k])), 0)
            emit, self.buf = self.buf[: len(self.buf) - keep], self.buf[len(self.buf) - keep :]
            if not self.inside:
                out.append(emit)
            return "".join(out)

    def flush(self) -> str:
        rest, self.buf = ("" if self.inside else self.buf), ""
        return rest


class ProviderError(RuntimeError):
    """`temporary=True`: rate-limited or busy; trying again a bit later should work.
    `model_unavailable=True`: this model is out (quota/long limit); another model may work."""

    def __init__(self, message: str, temporary: bool = False, model_unavailable: bool = False):
        super().__init__(message)
        self.temporary = temporary
        self.model_unavailable = model_unavailable


# Known free-tier request rates (seconds between requests to one provider). Anything not listed
# starts unpaced; every provider slows down automatically when it answers 429.
BASE_INTERVALS = {"mistral": 1.1}
MAX_INTERVAL = 10.0
MAX_RETRY_AFTER = 20.0


class Pacer:
    """Spaces out requests to one provider. On 429 the gap doubles (up to MAX_INTERVAL);
    each success eases it back toward the provider's normal rate."""

    def __init__(self, base: float = 0.0):
        self.base = base
        self.interval = base
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            loop = asyncio.get_running_loop()
            delay = self._next - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            self._next = loop.time() + self.interval

    def slow_down(self, retry_after: float | None = None) -> None:
        self.interval = min(MAX_INTERVAL, max(self.interval * 2, 1.0, retry_after or 0))
        if retry_after:
            self._next = max(self._next, asyncio.get_running_loop().time() + retry_after)

    def ok(self) -> None:
        self.interval = max(self.base, self.interval * 0.8)


def _retry_after(r: httpx.Response) -> float | None:
    value = r.headers.get("retry-after")
    try:
        return min(MAX_RETRY_AFTER, float(value)) if value else None
    except ValueError:
        return None


class OpenAICompatBackend:
    # Free tiers often answer "busy" (503) or "slow down" (429) for a few seconds; wait and retry.
    RETRYABLE = frozenset({429, 500, 502, 503, 504})
    RETRY_DELAYS = (2.0, 5.0)

    def __init__(self, client: httpx.AsyncClient | None = None, timeout: float = 120.0):
        self._client = client
        self.timeout = timeout
        self._pacers: dict[str, Pacer] = {}
        self._blocked: dict[tuple[str, str], float] = {}  # (provider, model) -> usable again at (monotonic)
        self._model_lists: dict[str, tuple[float, list[str]]] = {}  # provider -> (fetched at, ids)

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    def pacer(self, provider: str) -> Pacer:
        if provider not in self._pacers:
            self._pacers[provider] = Pacer(BASE_INTERVALS.get(provider, 0.0))
        return self._pacers[provider]

    def _request(self, agent: Agent, roster: list[Agent], user: str, **extra) -> tuple[ProviderSpec, dict]:
        spec = PROVIDERS[agent.provider]
        if not spec.url:
            raise ProviderError(f"{spec.label} has no base URL configured")
        body = {
            "model": agent.model_id,
            "messages": [
                {"role": "system", "content": agent.system_prompt(roster)},
                {"role": "user", "content": user},
            ],
            **extra,
        }
        return spec, adapt_body(agent.provider, body)

    def _note(self, provider: str, r: httpx.Response) -> None:
        """Feed every response into the provider's pacer."""
        if r.status_code == 429 and not quota_used_up(r):
            self.pacer(provider).slow_down(_retry_after(r))
        elif r.status_code < 400:
            self.pacer(provider).ok()

    def _block(self, agent: Agent, r: httpx.Response) -> None:
        wait = limit_wait(r)
        self._blocked[(agent.provider, agent.model_id)] = asyncio.get_running_loop().time() + (wait if wait else 6 * 3600)

    async def alternative_model(self, agent: Agent) -> str | None:
        """Another chat model from the same provider that isn't known to be out of quota.
        Each model has its own free quota, so when one runs out the agent can keep going."""
        spec = PROVIDERS[agent.provider]
        now = asyncio.get_running_loop().time()
        fetched, ids = self._model_lists.get(agent.provider, (0.0, []))
        if now - fetched > 600 or not ids:
            try:
                r = await self.client.get(f"{spec.url}/models", headers=auth_headers(spec), timeout=10)
                r.raise_for_status()
                ids = sorted(
                    m["id"].removeprefix("models/") for m in r.json().get("data", []) if isinstance(m, dict) and "id" in m
                )
                self._model_lists[agent.provider] = (now, ids)
            except (httpx.HTTPError, ValueError, KeyError):
                return None
        for model in candidate_models(agent.provider, spec.default_model, ids):
            if model == agent.model_id or self._blocked.get((agent.provider, model), 0) > now:
                continue
            return model
        return None

    def _delay(self, base: float, r: httpx.Response) -> float:
        return max(base, _retry_after(r) or 0)

    async def bid(self, agent: Agent, roster: list[Agent], transcript: list[Message], whiteboard: str = "") -> Bid:
        pacer = self.pacer(agent.provider)
        try:
            spec, body = self._request(
                agent, roster,
                _prompt(transcript, BID_INSTRUCTIONS.format(name=agent.name) + JSON_BID_SUFFIX, whiteboard),
                temperature=0.2, max_tokens=1024,  # roomy enough for models that think out loud first
            )
            for delay in (*self.RETRY_DELAYS[:1], None):  # bids are cheap: retry once, the room has a deadline
                await pacer.wait()
                r = await self.client.post(f"{spec.url}/chat/completions", json=body, headers=auth_headers(spec))
                self._note(agent.provider, r)
                if r.status_code not in self.RETRYABLE or delay is None or quota_used_up(r):
                    break
                await asyncio.sleep(self._delay(delay, r))
            if r.status_code == 429:
                if quota_used_up(r):
                    self._block(agent, r)
                    return _failed_bid("(daily quota used up)")
                return _failed_bid("(rate limited)")
            if r.status_code == 503:
                return _failed_bid("(provider busy)")
            r.raise_for_status()
            content = r.json()["choices"][0]["message"].get("content") or ""
        except (httpx.HTTPError, ProviderError, KeyError, IndexError, ValueError) as e:
            return _failed_bid(f"(error: {_describe(e)})")
        return parse_bid(content)

    async def speak(
        self, agent: Agent, roster: list[Agent], transcript: list[Message], whiteboard: str = ""
    ) -> AsyncIterator[str]:
        spec, body = self._request(
            agent, roster, _prompt(transcript, SPEAK_INSTRUCTIONS.format(name=agent.name), whiteboard),
            temperature=0.7, max_tokens=2048, stream=True,
        )
        pacer = self.pacer(agent.provider)
        think = ThinkFilter()
        for delay in (*self.RETRY_DELAYS, None):
            await pacer.wait()
            async with self.client.stream(
                "POST", f"{spec.url}/chat/completions", json=body, headers=auth_headers(spec)
            ) as r:
                if r.status_code >= 400:
                    await r.aread()
                self._note(agent.provider, r)
                if r.status_code == 429 and quota_used_up(r):
                    self._block(agent, r)
                    raise ProviderError(unavailable_reason(r, spec.label, agent.model_id) + ".", model_unavailable=True)
                if r.status_code in self.RETRYABLE and delay is not None:
                    wait = self._delay(delay, r)  # wait and retry below
                elif r.status_code in (429, 503):
                    what = "is rate-limiting requests" if r.status_code == 429 else "is overloaded"
                    raise ProviderError(f"{spec.label} {what} right now (free tiers allow only a few requests).", temporary=True)
                elif r.status_code >= 400:
                    raise ProviderError(f"{spec.label} returned {r.status_code}: {_error_text(r.text)}")
                else:
                    async for chunk in self._stream_text(r, think):
                        yield chunk
                    break
            await asyncio.sleep(wait)
        tail = think.flush()
        if tail:
            yield tail

    @staticmethod
    async def _stream_text(r: httpx.Response, think: "ThinkFilter") -> AsyncIterator[str]:
        async for line in r.aiter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                delta = json.loads(payload)["choices"][0].get("delta", {}).get("content")
            except (ValueError, KeyError, IndexError):
                continue
            if delta:
                text = think.feed(delta)
                if text:
                    yield text


_WAIT_RE = re.compile(
    r"(?:retry|try again)(?: after| in)?\s+((?:\d+(?:\.\d+)?h)?(?:\d+(?:\.\d+)?m(?!s))?(?:\d+(?:\.\d+)?s)?(?:\d+(?:\.\d+)?ms)?)"
)
SHORT_WAIT = 60.0  # seconds: pace and retry; longer means the model is out of action for a while


def limit_wait(r: httpx.Response) -> float | None:
    """How long the provider says to wait, from Retry-After or the message text
    ("Please try again in 7m12.5s", "retry in 41.2s", "try again in 450ms")."""
    header = r.headers.get("retry-after")
    try:
        if header:
            return float(header)
    except ValueError:
        pass
    m = _WAIT_RE.search(r.text.lower())
    if not m or not m.group(1):
        return None
    total = 0.0
    for value, unit in re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", m.group(1)):
        total += float(value) * {"h": 3600, "m": 60, "s": 1, "ms": 0.001}[unit]
    return total


def quota_used_up(r: httpx.Response) -> bool:
    """A 429 that a short wait won't fix: a daily quota, or a limit that clears only in
    minutes. Per-minute limits say "try again in 5s"; those are paced and retried instead.
    (Don't key on words like "billing": Groq's per-minute messages link to its billing page.)"""
    if r.status_code != 429:
        return False
    wait = limit_wait(r)
    if wait is not None:
        return wait > SHORT_WAIT
    text = r.text.lower()
    return any(k in text for k in ("quota", "per day", "perday", "(tpd)", "(rpd)", "daily"))


def unavailable_reason(r: httpx.Response, label: str, model: str) -> str:
    wait = limit_wait(r)
    text = r.text.lower()
    daily = any(k in text for k in ("per day", "perday", "(tpd)", "(rpd)", "daily")) or (wait or 0) > 3600
    when = "" if wait is None else (f" (resets in ~{wait / 3600:.0f}h)" if wait >= 3600 else f" (resets in ~{max(1, round(wait / 60))} min)")
    return f"{label}'s {'daily' if daily else 'free'} limit for {model} is reached{when}"


def _error_text(body: str) -> str:
    """Pull the human-readable message out of a provider's JSON error body."""
    try:
        data = json.loads(body)
        data = data[0] if isinstance(data, list) and data else data
        return str(data.get("error", {}).get("message") or body)[:300]
    except (ValueError, AttributeError):
        return body[:300]


def _describe(e: Exception) -> str:
    if isinstance(e, httpx.HTTPStatusError):
        code = e.response.status_code
        return {401: "key rejected", 403: "key not allowed", 404: "model not found"}.get(code, f"HTTP {code}")
    if isinstance(e, httpx.ConnectError):
        return "can't connect"
    if isinstance(e, httpx.TimeoutException):
        return "timed out"
    return type(e).__name__


# ---- routing & mock -------------------------------------------------------


class RoutingBackend:
    """Sends each agent to the backend for its provider."""

    def __init__(self, claude: Backend | None = None, compat: Backend | None = None):
        self.claude = claude or ClaudeBackend()
        self.compat = compat or OpenAICompatBackend()

    def _for(self, agent: Agent) -> Backend:
        return self.claude if agent.provider == "anthropic" else self.compat

    async def alternative_model(self, agent: Agent) -> str | None:
        finder = getattr(self._for(agent), "alternative_model", None)
        return await finder(agent) if finder else None

    async def bid(self, agent: Agent, roster: list[Agent], transcript: list[Message], whiteboard: str = "") -> Bid:
        return await self._for(agent).bid(agent, roster, transcript, whiteboard)

    def speak(
        self, agent: Agent, roster: list[Agent], transcript: list[Message], whiteboard: str = ""
    ) -> AsyncIterator[str]:
        return self._for(agent).speak(agent, roster, transcript, whiteboard)


class MockBackend:
    """Offline backend. Agents bid high when @mentioned, otherwise randomly-but-seeded,
    and get progressively quieter so conversations end on their own."""

    def __init__(self, seed: int = 7, delay: float = 0.01):
        self.rng = random.Random(seed)
        self.delay = delay

    async def bid(self, agent: Agent, roster: list[Agent], transcript: list[Message], whiteboard: str = "") -> Bid:
        await asyncio.sleep(self.delay)
        last = transcript[-1] if transcript else None
        if last and re.search(rf"@{re.escape(agent.name)}\b", last.text, re.IGNORECASE):
            return Bid(urgency=0.95, reason="I was mentioned")
        agent_turns = sum(1 for m in transcript if m.author != "Human")
        decay = max(0.0, 1.0 - 0.2 * agent_turns)
        return Bid(urgency=round(self.rng.random() * decay, 2), reason="mock bid")

    async def speak(
        self, agent: Agent, roster: list[Agent], transcript: list[Message], whiteboard: str = ""
    ) -> AsyncIterator[str]:
        last = transcript[-1] if transcript else None
        others = [a.name for a in roster if a.name != agent.name]
        handoff = f" @{self.rng.choice(others)}, thoughts?" if others and self.rng.random() < 0.3 else ""
        reply = (
            f"As the {agent.role} (mock {agent.model_id}), responding to "
            f"{last.author if last else 'nobody'}: here's my take.{handoff}"
        )
        # Exercise the tools when the human asks for them, so mock mode demos the whole UI.
        asked = last.text.lower() if last and last.author == "Human" else ""
        if "whiteboard" in asked and "whiteboard" in agent.tools:
            board = whiteboard.strip() or "# Plan"
            reply += f"\n```whiteboard\n{board}\n- [ ] {agent.name}: {agent.role} pass\n```"
        if ("search" in asked or "look up" in asked) and "search" in agent.tools:
            reply += f"\n[[search: {asked[:40].strip()}]]"
        for i in range(0, len(reply), 6):
            await asyncio.sleep(self.delay)
            yield reply[i : i + 6]
