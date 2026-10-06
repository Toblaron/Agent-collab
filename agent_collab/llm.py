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
from .providers import PROVIDERS, ProviderSpec, auth_headers

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
    pass


class OpenAICompatBackend:
    # Free tiers often answer "busy" (503) or "slow down" (429) for a few seconds; wait and retry.
    RETRYABLE = frozenset({429, 500, 502, 503, 504})
    RETRY_DELAYS = (1.5, 4.0)

    def __init__(self, client: httpx.AsyncClient | None = None, timeout: float = 120.0):
        self._client = client
        self.timeout = timeout

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

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
        return spec, body

    async def bid(self, agent: Agent, roster: list[Agent], transcript: list[Message], whiteboard: str = "") -> Bid:
        try:
            spec, body = self._request(
                agent, roster,
                _prompt(transcript, BID_INSTRUCTIONS.format(name=agent.name) + JSON_BID_SUFFIX, whiteboard),
                temperature=0.2, max_tokens=1024,  # roomy enough for models that think out loud first
            )
            for delay in (*self.RETRY_DELAYS[:1], None):  # bids are cheap: retry once, the room has a deadline
                r = await self.client.post(f"{spec.url}/chat/completions", json=body, headers=auth_headers(spec))
                if r.status_code not in self.RETRYABLE or delay is None:
                    break
                await asyncio.sleep(delay)
            if r.status_code == 429:
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
        think = ThinkFilter()
        for delay in (*self.RETRY_DELAYS, None):
            async with self.client.stream(
                "POST", f"{spec.url}/chat/completions", json=body, headers=auth_headers(spec)
            ) as r:
                if r.status_code in self.RETRYABLE and delay is not None:
                    await r.aread()
                elif r.status_code >= 400:
                    await r.aread()
                    raise ProviderError(f"{spec.label} returned {r.status_code}: {_error_text(r.text)}")
                else:
                    async for chunk in self._stream_text(r, think):
                        yield chunk
                    break
            await asyncio.sleep(delay)
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
        return f"HTTP {e.response.status_code}"
    if isinstance(e, httpx.ConnectError):
        return "can't connect"
    return type(e).__name__


# ---- routing & mock -------------------------------------------------------


class RoutingBackend:
    """Sends each agent to the backend for its provider."""

    def __init__(self, claude: Backend | None = None, compat: Backend | None = None):
        self.claude = claude or ClaudeBackend()
        self.compat = compat or OpenAICompatBackend()

    def _for(self, agent: Agent) -> Backend:
        return self.claude if agent.provider == "anthropic" else self.compat

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
