# agent-collab

A chat room where AI agents collaborate the way people do in a good working session: they
jump in when they have something to add, hand work to each other with `@mentions`, push back,
and go quiet when the job's done. You're in the room too, and your message interrupts whoever
is talking.

## How "natural" turn-taking works

Most multi-agent demos use round-robin (robotic) or a moderator LLM (bottleneck + extra call
per turn). This uses **bid-to-speak**:

1. After every message, each agent (except whoever just spoke) privately returns a structured
   bid: `{urgency: 0–1, reason}`. Bids run in parallel.
2. Scores are adjusted with a **dominance penalty** (−0.12 per message that agent posted in the
   last 4), so one voice can't hog the floor.
3. The top score above `speak_threshold` (0.35) speaks, streamed token-by-token to the UI.
4. When nobody clears the threshold, the room **goes quiet on its own**. `max_agent_turns` (12
   per human message) is the hard stop against runaway loops.
5. A human message cancels the current speaker mid-sentence (kept as `— (interrupted)`) and
   restarts the cycle.

The sidebar shows every agent's live bid, adjusted score and one-line reason, so you can see
*why* someone took the floor.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=sk-ant-...      # or `ant auth login`
agent-collab                              # http://127.0.0.1:8000
```

Rooms are keyed by URL hash: `http://127.0.0.1:8000/#launch-plan`. Several browser tabs on
the same room all see the same live conversation.

**No key / UI work:** `AGENT_COLLAB_MOCK=1 agent-collab` runs a deterministic offline backend.

## Configuration

| Env var | Default | Notes |
|---|---|---|
| `AGENT_COLLAB_MODEL` | `claude-opus-5-5` | Used for both bidding and speaking |
| `AGENT_COLLAB_BID_EFFORT` | `low` | Bids are cheap yes/no-ish calls |
| `AGENT_COLLAB_SPEAK_EFFORT` | `medium` | Raise to `high` for harder tasks |
| `AGENT_COLLAB_MOCK` | unset | `1` = offline mock backend |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | |

Requests use prompt caching (stable system prompt per agent), structured outputs for bids
(`messages.parse` + Pydantic), and server-side refusal fallbacks (`fallbacks="default"`).

**Cost note:** each agent turn costs N−1 bid calls + 1 speak call (4 agents → 3 bids + 1 reply).
Bids are short and low-effort, but if spend matters, point bids at a cheaper model (see
Roadmap).

## Layout

```
agent_collab/
  agents.py    Agent personas, Message, default roster (Ada/Bo/Cy/Dee)
  llm.py       Backend protocol, ClaudeBackend (bid + stream), MockBackend
  room.py      Room: bidding, scoring, speaking, interruption, pub/sub
  server.py    FastAPI app — GET /, GET /healthz, WS /ws/{room}
  static/      Single-file UI (no build step)
tests/         Turn-taking + WebSocket tests (no API calls)
```

WebSocket protocol — client sends `{"type":"say","text":...}` or `{"type":"stop"}`; server
emits `history`, `message`, `status`, `bids`, `stream_start`, `stream_delta`, `stream_end`.

## Customising the team

Edit `DEFAULT_ROSTER` in `agents.py`. Each `Agent` is a name, role, persona paragraph and
colour; the shared room etiquette lives in `Agent.system_prompt`.

## Roadmap

- Separate `AGENT_COLLAB_BID_MODEL` so bids can run on a cheaper model
- Tools per agent (web search, code execution, a shared scratchpad/whiteboard)
- Persistence (SQLite) and room/roster creation from the UI
- Private side-channels (agent ↔ agent DMs) and a task board agents can claim items from
- Long-room context management (compaction / rolling summary)

## Tests

```bash
pytest -q
```
