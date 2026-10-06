"""Start the app:  python -m agent_collab [demo|doctor]

Loads keys.env from the current folder. With any provider key (or Ollama running)
you get a real mixed-model team; otherwise the keyless demo. `demo` forces the demo,
`doctor` prints a setup report you can share (keys are masked).
"""

import os
import platform
import sys
from pathlib import Path

from .keysfile import load_keys, mask

PROVIDER_KEYS = ("GEMINI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY", "MISTRAL_API_KEY",
                 "HF_TOKEN", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "CUSTOM_LLM_BASE_URL")


def doctor(keys: dict, problems: list) -> None:
    from . import server
    from .providers import PROVIDERS

    print(f"agent-collab doctor · Python {platform.python_version()} · {platform.system()} {platform.machine()}")
    print(f"keys.env: {'found' if Path('keys.env').exists() else 'MISSING (cp keys.env.example keys.env)'} in {Path.cwd()}")
    for name, value in keys.items():
        print(f"  {name:<20} {mask(value)}")
    for p in problems:
        print(f"  !! {p}")
    print(f"Provider check (a 5-token test message each, up to {server.CHECK_DEADLINE:.0f}s):", flush=True)
    server.check_providers(on_result=lambda c: print(server.format_check(c), flush=True))
    print("Share this screen if something shows !! (keys above are masked).")


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    keys, problems = load_keys(Path("keys.env"))
    if mode == "doctor":
        doctor(keys, problems)
        return
    for p in problems:
        print(f"keys.env warning: {p}", flush=True)
    has_provider = any(os.environ.get(k) for k in PROVIDER_KEYS)
    if not has_provider:  # Ollama needs no key; is it running? (don't import .server yet: it reads the env)
        import httpx
        from .providers import PROVIDERS
        try:
            has_provider = httpx.get(f"{PROVIDERS['ollama'].url}/models", timeout=1.0).status_code == 200
        except httpx.HTTPError:
            pass
    if mode == "fast":
        os.environ["AGENT_COLLAB_SKIP_CHECK"] = "1"
    if mode == "demo" or not has_provider:
        os.environ["AGENT_COLLAB_MOCK"] = "1"
        print("Demo mode (mock agents). For real agents: cp keys.env.example keys.env, add a key, rerun.", flush=True)
    else:
        os.environ.setdefault("AGENT_COLLAB_DEFAULT_PROVIDER", "auto")
    from .server import main as serve
    serve()


main()
