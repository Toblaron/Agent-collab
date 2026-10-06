"""A chat room where agents bid for the floor.

Turn-taking: after every message, each agent (except the one who just spoke)
privately bids an urgency score. Scores are adjusted to stop any one agent
from dominating, the highest bid above the threshold speaks, and the room
goes quiet on its own once nobody has anything worth adding. A human message
interrupts whoever is talking and restarts the cycle; an @mention hands the
floor straight to the named agent.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass

from .actions import MAX_WHITEBOARD_CHARS, SEARCH, Actions, extract_actions
from .agents import Agent, AgentError, Message, agent_from_dict
from .llm import Backend, Bid
from .search import SearchError, Searcher, format_results

log = logging.getLogger(__name__)

HUMAN = "Human"
SEARCH_AUTHOR = "Search"
MAX_AGENTS = 8
MAX_TURNS_LIMIT = 50


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


# Bids that failed for a reason that fixes itself (worth waiting for) vs. one that won't.
TEMPORARY_FAILURES = ("(rate limited)", "(provider busy)", "(too slow", "(error: timed out", "(error: can't connect")
PERMANENT_FAILURES = ("(error", "(Claude not installed)", "(declined)", "(daily quota used up)")


def _failed(bid: Bid) -> bool:
    return bid.urgency == 0 and bid.reason.startswith(TEMPORARY_FAILURES + PERMANENT_FAILURES)


def _temporary(bid: Bid) -> bool:
    return bid.urgency == 0 and bid.reason.startswith(TEMPORARY_FAILURES)


def _mentions(name: str, text: str) -> re.Match | None:
    return re.search(rf"@{re.escape(name)}\b", text, re.IGNORECASE)


class Room:
    def __init__(
        self,
        room_id: str,
        agents: list[Agent],
        backend: Backend,
        settings: RoomSettings | None = None,
        searcher: Searcher | None = None,
        on_change: Callable[["Room"], None] | None = None,
    ):
        self.id = room_id
        self.agents = agents
        self.backend = backend
        self.settings = settings or RoomSettings()
        self.searcher = searcher
        self.on_change = on_change  # persistence hook: called after every durable change
        self.messages: list[Message] = []
        self.whiteboard = ""
        self.whiteboard_by: str | None = None
        self.muted: set[str] = set()  # benched by the human; they neither bid nor speak
        self.created_at = time.time()
        self.updated_at = self.created_at
        self._subscribers: set[asyncio.Queue] = set()
        self._task: asyncio.Task | None = None
        self._live: Message | None = None  # the reply being streamed right now (for clients that reconnect mid-reply)
        self._rng = random.Random()

    # ---- persistence ---------------------------------------------------

    @property
    def title(self) -> str:
        first = next((m.text for m in self.messages if m.author == HUMAN), "")
        first = " ".join(first.split())
        return (first[:60] + "…") if len(first) > 60 else (first or self.id)

    def to_state(self) -> dict:
        return {
            "version": 1,
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "agents": [a.to_dict() for a in self.agents],
            "muted": sorted(self.muted),
            "max_turns": self.settings.max_agent_turns,
            "whiteboard": self.whiteboard,
            "whiteboard_by": self.whiteboard_by,
            "messages": [m.to_dict() for m in self.messages],
        }

    def load_state(self, state: dict) -> None:
        agents: list[Agent] = []
        for data in state.get("agents") or []:
            with contextlib.suppress(AgentError, KeyError, TypeError, ValueError):
                agents.append(agent_from_dict(data, agents, require_ready=False))
        if agents:
            self.agents = agents
        self.messages = [Message.from_dict(m) for m in state.get("messages") or [] if isinstance(m, dict) and "author" in m]
        self.whiteboard = str(state.get("whiteboard") or "")
        self.whiteboard_by = state.get("whiteboard_by")
        self.muted = {n for n in state.get("muted") or [] if any(a.name == n for a in self.agents)}
        self.settings.max_agent_turns = _clamp_turns(state.get("max_turns", self.settings.max_agent_turns))
        self.created_at = float(state.get("created_at") or self.created_at)
        self.updated_at = float(state.get("updated_at") or self.updated_at)

    def _changed(self) -> None:
        self.updated_at = time.time()
        if self.on_change:
            try:
                self.on_change(self)
            except Exception:  # a full disk must never take the conversation down with it
                log.exception("saving room %s failed", self.id)

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
            "title": self.title,
            "agents": [a.to_dict() for a in self.agents],
            "muted": sorted(self.muted),
            "max_turns": self.settings.max_agent_turns,
            "messages": [m.to_dict() for m in self.messages],
            "running": self.running,
            "whiteboard": self.whiteboard,
            "whiteboard_by": self.whiteboard_by,
            "search": self.searcher.name if self.searcher else None,
            "live": self._live.to_dict() if self._live is not None else None,
        }

    # ---- control -------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def post_human(self, text: str) -> None:
        await self.stop()
        self._append(Message(author=HUMAN, text=text))
        self._start()

    async def continue_conversation(self) -> None:
        """Give the agents another round without a new human message (after the room went quiet)."""
        if self.running:
            return
        if not self.messages:
            raise AgentError("Nothing to continue yet: give the team a task first.")
        self._start(nudge=True)

    def _start(self, nudge: bool = False) -> None:
        self._task = asyncio.create_task(self._conversation_loop(nudge=nudge))

    async def stop(self) -> None:
        if self.running:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._task = None

    async def clear(self) -> None:
        """Wipe the conversation and whiteboard; the team stays."""
        await self.stop()
        self.messages = []
        self.whiteboard, self.whiteboard_by = "", None
        self._changed()
        self._emit(self.snapshot())

    def add_agent(self, data: dict) -> Agent:
        if len(self.agents) >= MAX_AGENTS:
            raise AgentError(f"A room holds at most {MAX_AGENTS} agents.")
        agent = agent_from_dict(data, self.agents)
        self.agents = [*self.agents, agent]  # new list: in-flight turns keep their snapshot
        self._roster_changed(f"{agent.name} ({agent.model_id}) joined the room")
        return agent

    def update_agent(self, name: str, data: dict) -> Agent:
        """Change an agent in place (model, persona, tools…). Renaming is allowed."""
        idx = next((i for i, a in enumerate(self.agents) if a.name == name), None)
        if idx is None:
            raise AgentError(f"No agent called {name}.")
        others = [a for a in self.agents if a.name != name]
        agent = agent_from_dict({**self.agents[idx].to_dict(), **data}, others)
        self.agents = [*self.agents[:idx], agent, *self.agents[idx + 1 :]]
        if name in self.muted:
            self.muted.discard(name)
            self.muted.add(agent.name)
        renamed = f" (now {agent.name})" if agent.name != name else ""
        self._roster_changed(f"{name}{renamed} updated: {agent.role}, {agent.model_id}")
        return agent

    def remove_agent(self, name: str) -> None:
        if not any(a.name == name for a in self.agents):
            raise AgentError(f"No agent called {name}.")
        self.agents = [a for a in self.agents if a.name != name]
        self.muted.discard(name)
        self._roster_changed(f"{name} left the room")

    def set_muted(self, name: str, muted: bool) -> None:
        agent = self._find(name)
        (self.muted.add if muted else self.muted.discard)(agent.name)
        self._roster_changed(f"{agent.name} {'is sitting out' if muted else 'is back'}")

    def set_max_turns(self, value) -> int:
        self.settings.max_agent_turns = _clamp_turns(value)
        self._changed()
        self._emit({"type": "settings", "max_turns": self.settings.max_agent_turns})
        return self.settings.max_agent_turns

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
        self.muted = set()
        note = f"Loaded team {label}: {', '.join(a.name for a in loaded)}"
        if skipped:
            note += f" (skipped {'; '.join(skipped)})"
        self._roster_changed(note)
        return skipped

    def set_whiteboard(self, text: str, by: str) -> None:
        self.whiteboard = text[:MAX_WHITEBOARD_CHARS]
        self.whiteboard_by = by
        self._changed()
        self._emit({"type": "whiteboard", "text": self.whiteboard, "by": by})

    def _find(self, name: str) -> Agent:
        agent = next((a for a in self.agents if a.name.lower() == name.lower().lstrip("@")), None)
        if agent is None:
            raise AgentError(f"No agent called {name}.")
        return agent

    def _roster_changed(self, note: str) -> None:
        self._changed()
        self._emit({"type": "roster", "agents": [a.to_dict() for a in self.agents], "muted": sorted(self.muted)})
        self._emit({"type": "notice", "text": note})

    async def wait_idle(self) -> None:
        if self._task:
            await self._task

    # ---- turn-taking ---------------------------------------------------

    def _append(self, message: Message) -> None:
        self.messages.append(message)
        self._changed()
        self._emit({"type": "message", "message": message.to_dict()})

    def _score(self, agent: Agent, bid: Bid) -> float:
        recent = self.messages[-self.settings.recent_window :]
        recent_turns = sum(1 for m in recent if m.author == agent.name)
        jitter = self._rng.uniform(0, 0.01)  # break exact ties without favouring roster order
        return bid.urgency - self.settings.dominance_penalty * recent_turns + jitter

    def _available(self, sitting_out: set[str]) -> list[Agent]:
        return [a for a in self.agents if a.name not in self.muted and a.name not in sitting_out]

    async def collect_bids(self, muted: set[str] = frozenset(), nudge: bool = False) -> list[ScoredBid]:
        last_author = self.messages[-1].author if self.messages else None
        # On a nudge ("continue") even the last speaker may go again: they may have more to say.
        bidders = [a for a in self._available(set(muted)) if nudge or a.name != last_author]
        transcript = list(self.messages)
        board = self.whiteboard
        last_text = self.messages[-1].text if self.messages else ""

        async def bid_with_deadline(agent: Agent) -> Bid:
            try:
                return await asyncio.wait_for(
                    self.backend.bid(self._as_prompted(agent), self.agents, transcript, board), self.settings.bid_timeout
                )
            except asyncio.TimeoutError:
                # A handoff ("@Bo, take a look") must not be lost just because Bo's provider is slow.
                if _mentions(agent.name, last_text):
                    return Bid(urgency=0.9, reason="(mentioned; bid timed out)")
                return Bid(urgency=0.0, reason="(too slow this round)")
            except Exception as e:  # a backend bug must not take the whole round down
                log.exception("bid from %s failed", agent.name)
                return Bid(urgency=0.0, reason=f"(error: {type(e).__name__})")

        bids = await asyncio.gather(*(bid_with_deadline(a) for a in bidders))
        scored = [ScoredBid(a, b, self._score(a, b)) for a, b in zip(bidders, bids)]
        return sorted(scored, key=lambda s: s.score, reverse=True)

    async def _conversation_loop(self, nudge: bool = False) -> None:
        self._emit({"type": "status", "state": "running"})
        sitting_out: set[str] = set()  # agents whose provider failed (or who said nothing) this round
        try:
            for turn in range(self.settings.max_agent_turns):
                mentioned = self._mentioned_agent(sitting_out)
                if mentioned is not None:
                    # A direct handoff: the named agent answers, no vote needed (and no N bid requests
                    # burned on free-tier rate limits).
                    bids = [ScoredBid(mentioned, Bid(urgency=1.0, reason="was @mentioned"), 1.0)]
                else:
                    first_nudge = nudge and turn == 0
                    self._emit({"type": "status", "state": "bidding"})
                    bids = await self.collect_bids(sitting_out, nudge=first_nudge)
                    if bids and all(_failed(b.bid) for b in bids) and not any(_temporary(b.bid) for b in bids):
                        # Bad keys / missing models: waiting won't help. Say what's wrong and stop.
                        reasons = "; ".join(f"{b.agent.name}: {b.bid.reason.strip('()')}" for b in bids)
                        self._emit({"type": "error", "agent": "room", "text": f"no agent could respond ({reasons}). Run `bash run.sh doctor` to check your keys."})
                        break
                    for wait in self.settings.throttle_waits:
                        if not bids or not all(_failed(b.bid) for b in bids):
                            break
                        self._emit({"type": "notice", "text": f"every agent's provider is rate-limited or busy; retrying in {wait:.0f}s"})
                        self._emit({"type": "status", "state": "waiting"})
                        await asyncio.sleep(wait)
                        self._emit({"type": "status", "state": "bidding"})
                        bids = await self.collect_bids(sitting_out, nudge=first_nudge)
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
                    actions, said = await self._speak(winner.agent)
                except asyncio.CancelledError:
                    raise
                except Exception as e:  # provider down, bad model ID, rate limit...
                    sitting_out.add(winner.agent.name)
                    self._emit({"type": "error", "agent": winner.agent.name, "text": f"{type(e).__name__}: {e}"[:300]})
                    continue
                if not said:
                    # An empty reply would leave the room unchanged and the same agent could win
                    # again and again; bench them for the rest of this round instead.
                    sitting_out.add(winner.agent.name)
                    self._emit({"type": "notice", "text": f"{winner.agent.name} had nothing to say"})
                    continue
                await self._apply(winner.agent, actions)
            else:
                if self.settings.max_agent_turns > 1:
                    self._emit(
                        {"type": "notice", "text": f"turn limit reached ({self.settings.max_agent_turns}); press continue for more"}
                    )
        except asyncio.CancelledError:
            raise
        except Exception as e:  # never die silently: show it and leave the room usable
            log.exception("conversation loop crashed in room %s", self.id)
            self._emit({"type": "error", "agent": "room", "text": f"{type(e).__name__}: {e}"[:300]})
        finally:
            self._emit({"type": "status", "state": "idle"})

    def _as_prompted(self, agent: Agent) -> Agent:
        """The agent as described to its model: never advertise a tool this server can't run
        (e.g. web search when it's turned off), or the model will keep trying to use it."""
        if self.searcher is None and SEARCH in agent.tools:
            return dataclasses.replace(agent, tools=tuple(t for t in agent.tools if t != SEARCH))
        return agent

    def _mentioned_agent(self, sitting_out: set[str]) -> Agent | None:
        """First agent @mentioned in the latest message (not its author, not benched)."""
        if not self.messages:
            return None
        last = self.messages[-1]
        hits = []
        for agent in self._available(sitting_out):
            if agent.name == last.author:
                continue
            match = _mentions(agent.name, last.text)
            if match:
                hits.append((match.start(), agent))
        return min(hits, key=lambda h: h[0])[1] if hits else None

    async def _speak(self, agent: Agent) -> tuple[Actions, bool]:
        """Stream one reply. Returns (actions to apply, whether anything was said)."""
        message = Message(author=agent.name, text="", meta={"model": agent.model_id, "provider": agent.provider})
        self._emit({"type": "stream_start", "id": message.id, "author": agent.name, "meta": message.meta})
        transcript = list(self.messages)
        actions = Actions()
        finished = False
        started = time.monotonic()
        self._live = message
        try:
            async for chunk in self.backend.speak(self._as_prompted(agent), self.agents, transcript, self.whiteboard):
                message.text += chunk
                self._emit({"type": "stream_delta", "id": message.id, "text": chunk})
            finished = True
        except asyncio.CancelledError:
            message.text = message.text.rstrip() + " — (interrupted)"
            raise
        finally:
            self._live = None
            message.meta["secs"] = round(time.monotonic() - started, 1)
            if finished:  # never act on half a message (interrupted or provider error)
                message.text, actions = extract_actions(message.text, agent.tools)
            message.text = message.text.strip()
            if message.text:
                self.messages.append(message)
                self._changed()
            self._emit({"type": "stream_end", "message": message.to_dict()})
        return actions, bool(message.text)

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
                except Exception as e:  # a searcher bug must not end the conversation
                    log.exception("search failed")
                    text = f'Search for "{query}" failed unexpectedly ({type(e).__name__}).'
            self._append(Message(author=SEARCH_AUTHOR, text=text))


def _clamp_turns(value) -> int:
    try:
        return max(1, min(MAX_TURNS_LIMIT, int(value)))
    except (TypeError, ValueError):
        return RoomSettings.max_agent_turns
