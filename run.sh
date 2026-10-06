#!/usr/bin/env bash
# Start the app. Usage:  bash run.sh        real agents if keys.env has any keys, else the demo
#                        bash run.sh demo   always the keyless demo (mock agents)
cd "$(dirname "$0")"

if [ -f keys.env ]; then
  set -a; . ./keys.env; set +a
  # Drop keys left blank so they don't count as "set".
  for v in $(grep -oE '^[A-Z_]+=' keys.env | tr -d '='); do
    [ -z "${!v}" ] && unset "$v"
  done
fi

has_key=""
for v in GEMINI_API_KEY GROQ_API_KEY OPENROUTER_API_KEY MISTRAL_API_KEY HF_TOKEN ANTHROPIC_API_KEY CUSTOM_LLM_BASE_URL; do
  [ -n "${!v}" ] && has_key=1
done
curl -s -m 1 http://localhost:11434/v1/models >/dev/null 2>&1 && has_key=1   # Ollama running

if [ "$1" = "demo" ] || [ -z "$has_key" ]; then
  export AGENT_COLLAB_MOCK=1
  echo "Demo mode (mock agents). For real agents: cp keys.env.example keys.env, add a key, rerun."
else
  export AGENT_COLLAB_DEFAULT_PROVIDER="${AGENT_COLLAB_DEFAULT_PROVIDER:-auto}"
fi
echo "Open http://127.0.0.1:8000 in your browser. Stop with CTRL+C."
exec python -m agent_collab
