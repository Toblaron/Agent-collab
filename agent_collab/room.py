"""A chat room where agents bid for the floor.

Turn-taking: after every message, each agent (except the one who just spoke)
privately bids an urgency score. Scores are adjusted to stop any one agent
from dominating, the highest bid above the threshold speaks, and the room
goes quiet on its own once nobody has anything worth adding. A human message
interrupts whoever is talking and restarts the cycle.
"""

from __future__ import annotations

import asyncio
import contextlib
import random
from dataclasses import dataclass

from .agents import Agent, Message
from .llm import Backend, Bid

HUMAN = "Human"


@dataclass
class RoomSettings:
    speak_threshold: float = 0.35  # minimum adjusted score to take the floor
    max_agent_turns: int = 12  # per human message; hard stop against runaway loops
    recent_window: int = 4  # how many recent messages count toward dominance penalty
    dominance_penalty: float = 0.12  # per recent message by the same agent


@dataclass
class ScoredBid:
    agent: Agent
    bid: Bid
    score: float

    def to_dict(self) -> dict:
        return {
            "agent": self.agent.name,
            "urgency": round(self.bid.urgency, 2),
            "score": round(self.score, 2),
            "reason": self.bid.reason,
        }


class Room:
    def __init__(self, room_id: str, agents: list[Agent], backend: Backend, settings: RoomSettings | None = None):
        self.id = room_id
        self.agents = agents
        self.backend = backend
        self.settings = settings or RoomSettings()
        self.messages: list[Message] = []
        self._subscribers: set[asyncio.Queue] = set()
        self._task: asyncio.Task | None = None
        self._rng = random.Random()

    # ---- pub/sub -------------------------------------------------------

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    def _emit(self, event: dict) -> None:
        for q in self._subscribers:
            q.put_nowait(event)

    def snapshot(self) -> dict:
        return {
            "type": "history",
            "room": self.id,
            "agents": [{"name": a.name, "role": a.role, "color": a.color} for a in self.agents],
            "messages": [m.to_dict() for m in self.messages],
            "running": self.running,
        }

    # ---- control -------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def post_human(self, text: str) -> None:
        await self.stop()
        self._append(Message(author=HUMAN, text=text))
        self._task = asyncio.create_task(self._conversation_loop())

    async def stop(self) -> None:
        if self.running:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._task = None

    async def wait_idle(self) -> None:
        if self._task:
            await self._task

    # ---- turn-taking ---------------------------------------------------

    def _append(self, message: Message) -> None:
        self.messages.append(message)
        self._emit({"type": "message", "message": message.to_dict()})

    def _score(self, agent: Agent, bid: Bid) -> float:
        recent = self.messages[-self.settings.recent_window :]
        recent_turns = sum(1 for m in recent if m.author == agent.name)
        jitter = self._rng.uniform(0, 0.01)  # break exact ties without favouring roster order
        return bid.urgency - self.settings.dominance_penalty * recent_turns + jitter

    async def collect_bids(self) -> list[ScoredBid]:
        last_author = self.messages[-1].author if self.messages else None
        bidders = [a for a in self.agents if a.name != last_author]
        transcript = list(self.messages)
        bids = await asyncio.gather(*(self.backend.bid(a, self.agents, transcript) for a in bidders))
        scored = [ScoredBid(a, b, self._score(a, b)) for a, b in zip(bidders, bids)]
        return sorted(scored, key=lambda s: s.score, reverse=True)

    async def _conversation_loop(self) -> None:
        self._emit({"type": "status", "state": "running"})
        try:
            for _ in range(self.settings.max_agent_turns):
                self._emit({"type": "status", "state": "bidding"})
                bids = await self.collect_bids()
                winner = bids[0] if bids and bids[0].score >= self.settings.speak_threshold else None
                self._emit(
                    {
                        "type": "bids",
                        "bids": [b.to_dict() for b in bids],
                        "winner": winner.agent.name if winner else None,
                    }
                )
                if winner is None:
                    break  # natural silence: nobody has anything worth adding
                await self._speak(winner.agent)
        finally:
            self._emit({"type": "status", "state": "idle"})

    async def _speak(self, agent: Agent) -> None:
        message = Message(author=agent.name, text="")
        self._emit({"type": "stream_start", "id": message.id, "author": agent.name})
        transcript = list(self.messages)
        try:
            async for chunk in self.backend.speak(agent, self.agents, transcript):
                message.text += chunk
                self._emit({"type": "stream_delta", "id": message.id, "text": chunk})
        except asyncio.CancelledError:
            message.text = message.text.rstrip() + " — (interrupted)"
            raise
        finally:
            message.text = message.text.strip()
            if message.text:
                self.messages.append(message)
            self._emit({"type": "stream_end", "message": message.to_dict()})
