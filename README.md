# agent-collab


A chat room where AI agents collaborate the way people do in a good working session: they
jump in when they have something to add, hand work to each other with `@mentions`, push back,
and go quiet when the job's done. You're in the room too, and your message interrupts whoever
is talking.

Agents don't have to run on the same model. Put Claude, a local Llama on Ollama, Gemini and a
free OpenRouter model in one room and let them work it out together. They share a whiteboard,
can search the web, and you can save a good team and bring it back later.

The UI is a terminal: Matrix-green on black with digital rain and CRT scanlines by default,
`[theme]` for a light variant, `[rain]` to switch the animation off (it's off automatically if
your OS asks for reduced motion).

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
pip install -e ".[dev,claude]"           # drop ",claude" if you only use free models
export ANTHROPIC_API_KEY=sk-ant-...      # or `ant auth login`
agent-collab                              # http://127.0.0.1:8000
```

Rooms are keyed by URL hash: `http://127.0.0.1:8000/#launch-plan`. Several browser tabs on
the same room all see the same live conversation.

**No key / UI work:** `AGENT_COLLAB_MOCK=1 agent-collab` runs a deterministic offline backend.

## Mixing in free LLMs

Click **+ Add agent** in the sidebar and pick a provider, a model (the list is fetched live from
the provider), a role and a personality. Remove an agent with **×**. Every agent's model is
shown on its messages, and agents are told which model each teammate runs on.

| Provider | Cost | Enable it with | Example model |
|---|---|---|---|
| **Ollama** (local) | Free, your hardware | Nothing: install [Ollama](https://ollama.com), `ollama pull llama3.2`. `OLLAMA_BASE_URL` for another host | `llama3.2`, `qwen3`, `mistral` |
| **Groq** | Free tier, rate-limited | `GROQ_API_KEY` | `llama-3.3-70b-versatile` |
| **Google Gemini** | Free tier, rate-limited | `GEMINI_API_KEY` (aistudio.google.com) | `gemini-flash-latest` |
| **OpenRouter** | Models ending in `:free` are free | `OPENROUTER_API_KEY` | `meta-llama/llama-3.3-70b-instruct:free` |
| **Hugging Face** | Free monthly credits | `HF_TOKEN` | `meta-llama/Llama-3.1-8B-Instruct` |
| **Mistral** | Free experiment tier | `MISTRAL_API_KEY` | `mistral-small-latest` |
| **Custom** | — | `CUSTOM_LLM_BASE_URL` (+ optional `CUSTOM_LLM_API_KEY`, `CUSTOM_LLM_MODEL`) | LM Studio, vLLM, llama.cpp server… |
| **Claude** | Paid API | `ANTHROPIC_API_KEY` or `ant auth login` | `claude-opus-5-5` |

Set the env vars, restart, and unconfigured providers light up in the picker. Free-tier models,
IDs and limits change often; the live model list in the dialog is the source of truth, and the
examples above are just defaults.

**Run the whole room for free:** move the starter team off Claude:

```bash
AGENT_COLLAB_DEFAULT_PROVIDER=ollama AGENT_COLLAB_DEFAULT_MODEL=llama3.2 agent-collab
```

How non-Claude agents work:

- **One adapter for all of them.** Every free provider speaks the OpenAI-compatible
  `/chat/completions` API, so `OpenAICompatBackend` covers them all; `RoutingBackend` sends each
  agent to Claude or to that adapter based on its provider.
- **Lenient bids.** Not every free model supports JSON mode, so bids are requested as JSON and
  parsed leniently (code fences, chatter, bare `urgency: 0.6` all work). Anything unparseable
  counts as "stay quiet".
- **Reasoning models.** `<think>…</think>` blocks (DeepSeek-R1, Qwen3…) are stripped from both
  bids and streamed replies, even when the tags are split across stream chunks.
- **Failure isolation.** A rate limit, a wrong model ID or Ollama not running mutes that agent for
  the rest of the round and shows a red error line; everyone else keeps talking. Bid failures show
  up as the agent's reason in the sidebar (`(rate limited)`, `(error: can't connect)`).
- **No arbitrary URLs from the browser.** Provider base URLs come only from the built-in table or
  env vars, so the UI can't be used to make the server call arbitrary hosts.

## Tools: shared whiteboard & web search

Tools work for **every** model, including small local ones with no function-calling support,
because they're plain-text markers the room parses after an agent finishes speaking:

````
```whiteboard
# Launch plan
- [ ] Bo: landing page
```

[[search: competitor pricing for team chat apps]]
````

- **Whiteboard (BOARD tab):** one shared document per room, shown to every agent on every turn.
  A whiteboard block replaces the whole document (agents are told to write the full updated
  version). You can edit it too: `[edit]`. In the chat the block collapses to
  `[updated the whiteboard]`.
- **Web search:** up to 2 queries per message. Results are posted to the room as a `search`
  message, so the whole team sees them and anyone (usually the asker) can pick them up on the
  next turn. Results are labelled untrusted web content in every agent's prompt.
- Tools are per agent: tick/untick them in the **+ agent** dialog. Markers from an agent
  without that tool are left as plain text.

| Search backend | Cost | Enable |
|---|---|---|
| DuckDuckGo | Free, no key (best-effort HTML scrape; can be rate-limited) | default when nothing else is set |
| Tavily | Free tier | `TAVILY_API_KEY` |
| Brave Search | Free tier | `BRAVE_API_KEY` |
| SearXNG | Free, self-hosted (enable JSON output) | `SEARXNG_URL` |

`AGENT_COLLAB_SEARCH=auto` (default) picks the first configured of Tavily → Brave → SearXNG →
DuckDuckGo. Force one with `AGENT_COLLAB_SEARCH=brave` etc., or `off`.

## Rooms, history & everyday use

- **Everything is saved.** Each room's conversation, whiteboard, team, muted agents and turn
  limit live in `~/.agent-collab/rooms/<room>.json` and come back after a restart. Empty rooms
  aren't saved. **ROOMS** tab: open, export or delete rooms, or start a new one.
- **Export** (`[export]` or `/export`): the conversation plus final whiteboard as Markdown.
- **[▶ continue]**: give the agents another round when the room has gone quiet or hit the
  turn limit, without typing anything.
- **Agents**: `[edit]` an agent in place (model, persona, tools, even name), `[mute]` to bench
  it without removing it, and set *agent turns per message* (1–50).
- **Commands** (type `/` for the list): `/continue /stop /clear /export /new [name]
  /room name /mute name /unmute name /turns N /board /theme /rain /help`. Start a message
  with `//` to send a literal slash.
- **@mentions** autocomplete as you type and hand the floor straight to that agent.
- Replies render Markdown (lists, tables, code, links), show the model and response time,
  and have a copy button. Visible scrollbars everywhere (including phones, where the browser's
  own are invisible), a jump-to-latest button, and an unread count in the tab title.
- Keys: `Enter` send, `Shift+Enter` newline, `Esc` stop the agents, `/` focus the input.
- Long rooms stay fast: agents see the latest 40 messages plus the whiteboard.

## Saved teams

TEAMS tab → name it → `[save]` stores the room's current roster (names, roles, personas,
providers, models, tools) as JSON in `~/.agent-collab/teams/` (`AGENT_COLLAB_DATA_DIR` to move
it). `[load]` swaps a saved team into any room; agents whose provider isn't configured on the
current server are skipped with a note, so a team file is safe to share between machines. Team
files are plain JSON: commit them, hand-edit them, share them.

## Configuration

| Env var | Default | Notes |
|---|---|---|
| `AGENT_COLLAB_MODEL` | `claude-opus-5-5` | Default Claude model (bidding and speaking) |
| `AGENT_COLLAB_DEFAULT_PROVIDER` | `anthropic` | Provider for the starter team (`ollama`, `groq`, …), or `auto` to spread it across every provider you've set up |
| `AGENT_COLLAB_DEFAULT_MODEL` | provider default | Model for the starter team |
| `AGENT_COLLAB_BID_EFFORT` | `low` | Claude only. Bids are cheap yes/no-ish calls |
| `AGENT_COLLAB_SPEAK_EFFORT` | `medium` | Claude only. Raise to `high` for harder tasks |
| `AGENT_COLLAB_SEARCH` | `auto` | `auto`, `off`, `tavily`, `brave`, `searxng`, `duckduckgo` |
| `AGENT_COLLAB_DATA_DIR` | `~/.agent-collab` | Where saved teams live |
| `AGENT_COLLAB_MOCK` | unset | `1` = offline mock LLMs + mock search (UI work, demos) |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | |

Claude requests use prompt caching (stable system prompt per agent), structured outputs for bids
(`messages.parse` + Pydantic), and server-side refusal fallbacks (`fallbacks="default"`).

**Cost note:** each agent turn costs N−1 bid calls + 1 speak call (4 agents → 3 bids + 1 reply).
Bids are short and low-effort, but if spend matters, point bids at a cheaper model (see
Roadmap).

## Layout

```
agent_collab/
  agents.py    Agent personas (+ provider/model/tools), validation, default roster (Ada/Bo/Cy/Dee)
  actions.py   Text-marker tool protocol: whiteboard blocks, [[search: …]]
  search.py    Search backends (Tavily, Brave, SearXNG, DuckDuckGo, mock)
  teams.py     Saved-team store (JSON files)
  providers.py Provider table (Claude, Ollama, Groq, Gemini, OpenRouter, HF, Mistral, custom)
  llm.py       ClaudeBackend, OpenAICompatBackend, RoutingBackend, MockBackend
  room.py      Room: bidding, speaking, tools, whiteboard, mute/continue/clear, pub/sub
  rooms.py     Saved rooms (JSON per room) + Markdown export
  server.py    FastAPI app: GET /, /healthz, /api/{config,teams,providers[/{id}/models]}, WS /ws/{room}
  static/      Single-file terminal UI (no build step)
tests/         Turn-taking, providers, tools, search parsers, teams, WebSocket (no network)
```

HTTP: `GET /api/rooms`, `GET /api/rooms/{id}/export.md`, `DELETE /api/rooms/{id}`,
`GET /api/teams`, `GET /api/providers[/{id}/models]`, `GET /api/config`.

WebSocket `/ws/{room}`: client sends `say`, `stop`, `continue`, `clear`, `add_agent` /
`update_agent` (`{"name", "agent": {name, role, provider, model, persona, color, tools}}`),
`remove_agent`, `set_muted` (`{"name", "muted"}`), `set_max_turns` (`{"value"}`),
`set_whiteboard` (`{"text"}`), `save_team` / `load_team` / `delete_team` (`{"name"}`).
Server emits `history` (includes any reply still streaming, for reconnects), `roster`,
`settings`, `message`, `status`, `bids`, `stream_start` / `stream_delta` / `stream_end`,
`whiteboard`, `teams`, `notice`, `error`, `deleted`, and `agent_error` (sender only).

## Customising the team

Add agents from the UI, or edit `DEFAULT_ROSTER` in `agents.py` (each `Agent` takes an optional
`provider=` and `model=`). The shared room etiquette lives in `Agent.system_prompt`.

## Roadmap

- Separate bid model per agent (e.g. bid on a small local model, speak on a big one)
- More tools: sandboxed code execution, fetch-a-URL
- Automatic model fallback when a free model's daily quota runs out mid-conversation
- Private side-channels (agent ↔ agent DMs) and a task board agents can claim items from
- Long-room context management (compaction / rolling summary)

## Tests

```bash
pytest -q
```

## Running on Android (Termux)

```bash
cd ~/matrix-os/Agent-collab     # wherever you cloned it
bash setup.sh                   # installs Rust + the app; first build takes 5-20 min
bash run.sh                     # then open http://127.0.0.1:8000 in Chrome
```

For real agents, put your keys in a file once: `cp keys.env.example keys.env`, then
`nano keys.env` and paste keys after the `=` signs (links to each free signup are in the file).
`bash run.sh` then builds a mixed starter team across every provider you have a key for;
without keys it starts the demo (`bash run.sh demo` forces the demo).
Something not working? `bash run.sh doctor` checks your keys (masked, safe to share) and
tests every provider without spending tokens. Keep Termux open while you use the app. If you paste
commands from a chat app, watch for curly quotes: a `>` prompt means bash is waiting for a
closing quote; press CTRL+C and retype it.
