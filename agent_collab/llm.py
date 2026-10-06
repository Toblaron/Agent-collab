"""LLM backends: how an agent bids for the floor and how it speaks.

`ClaudeBackend` calls the Anthropic API. `MockBackend` is deterministic and
offline, used by tests and by `AGENT_COLLAB_MOCK=1` for UI work without spend.
"""

from __future__ import annotations

import asyncio
import os
import random
import re
from typing import AsyncIterator, Protocol

import anthropic
from pydantic import BaseModel, Field

from .agents import Agent, Message, render_transcript

MODEL = os.environ.get("AGENT_COLLAB_MODEL", "claude-opus-5-5")
BID_EFFORT = os.environ.get("AGENT_COLLAB_BID_EFFORT", "low")
SPEAK_EFFORT = os.environ.get("AGENT_COLLAB_SPEAK_EFFORT", "medium")
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class Bid(BaseModel):
    """An agent's answer to 'do you want to speak next?'"""

    urgency: float = Field(description="0.0 = nothing to add, 1.0 = must speak right now")
    reason: str = Field(description="One short phrase: what you would say, or why you are staying quiet")


class Backend(Protocol):
    async def bid(self, agent: Agent, roster: list[Agent], transcript: list[Message]) -> Bid: ...

    def speak(self, agent: Agent, roster: list[Agent], transcript: list[Message]) -> AsyncIterator[str]: ...


BID_INSTRUCTIONS = (
    "Above is the conversation so far. Decide whether YOU should speak next.\n"
    "Score urgency from 0.0 to 1.0:\n"
    "- 0.8-1.0: you were directly addressed (@{name}), or you see a mistake or blocker no one has raised\n"
    "- 0.5-0.7: you have a genuinely new contribution that moves the work forward\n"
    "- 0.1-0.4: you could add something minor\n"
    "- 0.0: you would only be agreeing, repeating, or the task is done\n"
    "Silence is a good default. Do not bid high just to participate."
)

SPEAK_INSTRUCTIONS = "Above is the conversation so far. It is your turn. Write only your next message, as {name}, with no name prefix."


def _user_turn(transcript: list[Message], instructions: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": f"<transcript>\n{render_transcript(transcript)}\n</transcript>\n\n{instructions}",
        }
    ]


class ClaudeBackend:
    def __init__(self, client: anthropic.AsyncAnthropic | None = None, model: str = MODEL):
        self._client = client
        self.model = model

    @property
    def client(self) -> anthropic.AsyncAnthropic:
        # Lazy so importing the app (tests, mock mode) never needs credentials.
        if self._client is None:
            self._client = anthropic.AsyncAnthropic()
        return self._client

    async def bid(self, agent: Agent, roster: list[Agent], transcript: list[Message]) -> Bid:
        try:
            response = await self.client.beta.messages.parse(
                model=self.model,
                max_tokens=4000,
                system=agent.system_prompt(roster),
                messages=_user_turn(transcript, BID_INSTRUCTIONS.format(name=agent.name)),
                output_format=Bid,
                output_config={"effort": BID_EFFORT},
                cache_control={"type": "ephemeral"},
                betas=[FALLBACK_BETA],
                fallbacks="default",
            )
        except anthropic.APIError:
            # A failed bid means "stay quiet this round", not a crashed room.
            return Bid(urgency=0.0, reason="(bid failed)")
        if response.stop_reason == "refusal" or response.parsed_output is None:
            return Bid(urgency=0.0, reason="(declined)")
        bid = response.parsed_output
        bid.urgency = max(0.0, min(1.0, bid.urgency))
        return bid

    async def speak(self, agent: Agent, roster: list[Agent], transcript: list[Message]) -> AsyncIterator[str]:
        async with self.client.beta.messages.stream(
            model=self.model,
            max_tokens=16000,
            system=agent.system_prompt(roster),
            messages=_user_turn(transcript, SPEAK_INSTRUCTIONS.format(name=agent.name)),
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


class MockBackend:
    """Offline backend. Agents bid high when @mentioned, otherwise randomly-but-seeded,
    and get progressively quieter so conversations end on their own."""

    def __init__(self, seed: int = 7, delay: float = 0.01):
        self.rng = random.Random(seed)
        self.delay = delay

    async def bid(self, agent: Agent, roster: list[Agent], transcript: list[Message]) -> Bid:
        await asyncio.sleep(self.delay)
        last = transcript[-1] if transcript else None
        if last and re.search(rf"@{re.escape(agent.name)}\b", last.text, re.IGNORECASE):
            return Bid(urgency=0.95, reason="I was mentioned")
        agent_turns = sum(1 for m in transcript if m.author != "Human")
        decay = max(0.0, 1.0 - 0.2 * agent_turns)
        return Bid(urgency=round(self.rng.random() * decay, 2), reason="mock bid")

    async def speak(self, agent: Agent, roster: list[Agent], transcript: list[Message]) -> AsyncIterator[str]:
        last = transcript[-1] if transcript else None
        others = [a.name for a in roster if a.name != agent.name]
        handoff = f" @{self.rng.choice(others)}, thoughts?" if others and self.rng.random() < 0.3 else ""
        reply = f"As the {agent.role}, responding to {last.author if last else 'nobody'}: here's my take.{handoff}"
        for word in reply.split(" "):
            await asyncio.sleep(self.delay)
            yield word + " "
