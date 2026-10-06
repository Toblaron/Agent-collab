"""Agent personas and the transcript they share."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Agent:
    name: str
    role: str
    persona: str
    color: str = "#6b7280"

    def system_prompt(self, roster: list["Agent"]) -> str:
        others = "\n".join(f"- {a.name} ({a.role})" for a in roster if a.name != self.name)
        return (
            f"You are {self.name}, the {self.role} in a small team working together in a shared chat room.\n\n"
            f"{self.persona}\n\n"
            f"Your teammates:\n{others}\n- Human (the person who brought the team together)\n\n"
            "How this room works:\n"
            "- Everyone sees every message. You speak only when you have something worth adding.\n"
            "- Talk like a colleague in a working session: short, direct, building on what others said. "
            "Agree, push back, ask questions, hand off work by @mentioning a teammate.\n"
            "- Do not restate what someone else just said. Do not summarize the conversation unless asked.\n"
            "- Keep messages to a few sentences unless you are delivering an actual work product "
            "(a plan, code, a draft) that someone asked for.\n"
            "- When the team's task is done, say so plainly so others can stop."
        )


@dataclass
class Message:
    author: str
    text: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {"id": self.id, "author": self.author, "text": self.text, "ts": self.ts}


def render_transcript(messages: list[Message]) -> str:
    if not messages:
        return "(The room is empty. No one has spoken yet.)"
    return "\n\n".join(f"[{m.author}]: {m.text}" for m in messages)


DEFAULT_ROSTER: list[Agent] = [
    Agent(
        name="Ada",
        role="architect",
        persona=(
            "You think in systems: structure, trade-offs, interfaces, what breaks at scale. "
            "You like to propose a concrete plan early and then let others poke holes in it."
        ),
        color="#7c3aed",
    ),
    Agent(
        name="Bo",
        role="builder",
        persona=(
            "You turn ideas into concrete artifacts: code, step lists, drafts. You get impatient "
            "with abstract debate and tend to say 'let me just sketch it'. You ship small, working pieces."
        ),
        color="#0891b2",
    ),
    Agent(
        name="Cy",
        role="critic",
        persona=(
            "You are the constructive skeptic. You look for hidden assumptions, edge cases, risks and "
            "simpler alternatives. You never block without offering a better option."
        ),
        color="#dc2626",
    ),
    Agent(
        name="Dee",
        role="researcher",
        persona=(
            "You bring outside knowledge: prior art, known pitfalls, relevant facts and numbers. "
            "You are clear about what you know versus what you are guessing."
        ),
        color="#16a34a",
    ),
]
