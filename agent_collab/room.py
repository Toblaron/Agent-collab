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
import re
from dataclasses import dataclass

from .actions import MAX_WHITEBOARD_CHARS, Actions, extract_actions
from .agents import Agent, AgentError, Message, agent_from_dict
from .llm import Backend, Bid
from .search import SearchError, Searcher, format_results

HUMAN = "Human"
SEARCH_AUTHOR = "Search"
MAX_AGENTS = 8


@dataclass
class RoomSettings:
    speak_threshold: float = 0.35  # minimum adjusted score to take the floor
    max_agent_turns: int = 12  # per human message; hard stop against runaway loops
    recent_window: int = 4  # how many recent messages count toward dominance penalty
    dominance_penalty: float = 0.12  # per recent message by the same agent
    bid_timeout: float = 15.0  # seconds; a slow provider sits the round out instead of stalling everyone
    throttle_waits: tuple[float, ...] = (15.0, 30.0)  # when *every* bid failed (rate limits), wait and retry


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


FAILED_BID_MARKERS = ("(rate limited)", "(provider busy)", "(error", "(too slow", "(Claude not installed)")


def _failed(bid: Bid) -> bool:
    return bid.urgency == 0 and bid.reason.startswith(FAILED_BID_MARKERS)


class Room:
    def __init__(
        self,
        room_id: str,
        agents: list[Agent],
        backend: Backend,
        settings: RoomSettings | None = None,
        searcher: Searcher | None = None,
    ):
        self.id = room_id
        self.agents = agents
        self.backend = backend
        self.settings = settings or RoomSettings()
        self.searcher = searcher
        self.messages: list[Message] = []
        self.whiteboard = ""
        self.whiteboard_by: str | None = None
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
            "agents": [a.to_dict() for a in self.agents],
            "messages": [m.to_dict() for m in self.messages],
            "running": self.running,
            "whiteboard": self.whiteboard,
            "whiteboard_by": self.whiteboard_by,
            "search": self.searcher.name if self.searcher else None,
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

    def add_agent(self, data: dict) -> Agent:
        if len(self.agents) >= MAX_AGENTS:
            raise AgentError(f"A room holds at most {MAX_AGENTS} agents.")
        agent = agent_from_dict(data, self.agents)
        self.agents = [*self.agents, agent]  # new list: in-flight turns keep their snapshot
        self._emit_roster()
        self._emit({"type": "notice", "text": f"{agent.name} ({agent.model_id}) joined the room"})
        return agent

    def remove_agent(self, name: str) -> None:
        if not any(a.name == name for a in self.agents):
            raise AgentError(f"No agent called {name}.")
        self.agents = [a for a in self.agents if a.name != name]
        self._emit_roster()
        self._emit({"type": "notice", "text": f"{name} left the room"})

    def load_roster(self, agent_dicts: list[dict], label: str) -> list[str]:
        """Replace the roster with a saved team. Agents that can't run here (e.g. their
        provider isn't configured on this server) are skipped; returns the reasons."""
        loaded: list[Agent] = []
        skipped: list[str] = []
        for data in agent_dicts[:MAX_AGENTS]:
            try:
                loaded.append(agent_from_dict(data, loaded))
            except AgentError as e:
                skipped.append(f"{data.get('name', '?')}: {e}")
        if not loaded:
            raise AgentError(f"None of the agents in {label} can run here. " + " ".join(skipped))
        self.agents = loaded
        self._emit_roster()
        note = f"Loaded team {label}: {', '.join(a.name for a in loaded)}"
        if skipped:
            note += f" (skipped {'; '.join(skipped)})"
        self._emit({"type": "notice", "text": note})
        return skipped

    def set_whiteboard(self, text: str, by: str) -> None:
        self.whiteboard = text[:MAX_WHITEBOARD_CHARS]
        self.whiteboard_by = by
        self._emit({"type": "whiteboard", "text": self.whiteboard, "by": by})

    def _emit_roster(self) -> None:
        self._emit({"type": "roster", "agents": [a.to_dict() for a in self.agents]})

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

    async def collect_bids(self, muted: set[str] = frozenset()) -> list[ScoredBid]:
        last_author = self.messages[-1].author if self.messages else None
        bidders = [a for a in self.agents if a.name != last_author and a.name not in muted]
        transcript = list(self.messages)
        board = self.whiteboard
        last_text = self.messages[-1].text if self.messages else ""

        async def bid_with_deadline(agent: Agent) -> Bid:
            try:
                return await asyncio.wait_for(
                    self.backend.bid(agent, self.agents, transcript, board), self.settings.bid_timeout
                )
            except asyncio.TimeoutError:
                # A handoff ("@Bo, take a look") must not be lost just because Bo's provider is slow.
                if re.search(rf"@{re.escape(agent.name)}\b", last_text, re.IGNORECASE):
                    return Bid(urgency=0.9, reason="(mentioned; bid timed out)")
                return Bid(urgency=0.0, reason="(too slow this round)")

        bids = await asyncio.gather(*(bid_with_deadline(a) for a in bidders))
        scored = [ScoredBid(a, b, self._score(a, b)) for a, b in zip(bidders, bids)]
        return sorted(scored, key=lambda s: s.score, reverse=True)

    async def _conversation_loop(self) -> None:
        self._emit({"type": "status", "state": "running"})
        muted: set[str] = set()  # agents whose provider failed this round
        try:
            for _ in range(self.settings.max_agent_turns):
                mentioned = self._mentioned_agent(muted)
                if mentioned is not None:
                    # A direct handoff: the named agent answers, no vote needed (and no N bid requests
                    # burned on free-tier rate limits).
                    bids = [ScoredBid(mentioned, Bid(urgency=1.0, reason="was @mentioned"), 1.0)]
                else:
                    self._emit({"type": "status", "state": "bidding"})
                    bids = await self.collect_bids(muted)
                    for wait in self.settings.throttle_waits:
                        if not bids or not all(_failed(b.bid) for b in bids):
                            break
                        self._emit({"type": "notice", "text": f"every agent's provider is rate-limited or busy; retrying in {wait:.0f}s"})
                        self._emit({"type": "status", "state": "waiting"})
                        await asyncio.sleep(wait)
                        self._emit({"type": "status", "state": "bidding"})
                        bids = await self.collect_bids(muted)
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
                try:
                    actions = await self._speak(winner.agent)
                except Exception as e:  # provider down, bad model ID, rate limit...
                    muted.add(winner.agent.name)
                    self._emit({"type": "error", "agent": winner.agent.name, "text": f"{type(e).__name__}: {e}"[:300]})
                    continue
                await self._apply(winner.agent, actions)
        finally:
            self._emit({"type": "status", "state": "idle"})

    def _mentioned_agent(self, muted: set[str]) -> Agent | None:
        """First agent @mentioned in the latest message (not its author, not muted)."""
        if not self.messages:
            return None
        last = self.messages[-1]
        hits = []
        for agent in self.agents:
            if agent.name == last.author or agent.name in muted:
                continue
            match = re.search(rf"@{re.escape(agent.name)}\b", last.text, re.IGNORECASE)
            if match:
                hits.append((match.start(), agent))
        return min(hits, key=lambda h: h[0])[1] if hits else None

    async def _speak(self, agent: Agent) -> Actions:
        message = Message(author=agent.name, text="")
        self._emit({"type": "stream_start", "id": message.id, "author": agent.name})
        transcript = list(self.messages)
        actions = Actions()
        finished = False
        try:
            async for chunk in self.backend.speak(agent, self.agents, transcript, self.whiteboard):
                message.text += chunk
                self._emit({"type": "stream_delta", "id": message.id, "text": chunk})
            finished = True
        except asyncio.CancelledError:
            message.text = message.text.rstrip() + " — (interrupted)"
            raise
        finally:
            if finished:  # never act on half a message (interrupted or provider error)
                message.text, actions = extract_actions(message.text, agent.tools)
            message.text = message.text.strip()
            if message.text:
                self.messages.append(message)
            self._emit({"type": "stream_end", "message": message.to_dict()})
        return actions

    async def _apply(self, agent: Agent, actions: Actions) -> None:
        if actions.whiteboard is not None:
            self.set_whiteboard(actions.whiteboard, by=agent.name)
        for query in actions.searches:
            if self.searcher is None:
                text = f'Search is turned off on this server, so "{query}" was not searched.'
            else:
                self._emit({"type": "status", "state": "searching"})
                try:
                    text = format_results(query, await self.searcher.search(query))
                except SearchError as e:
                    text = f'Search for "{query}" failed: {e}'
            self._append(Message(author=SEARCH_AUTHOR, text=text))
