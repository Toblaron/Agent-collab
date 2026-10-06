#!/usr/bin/env bash
# Start the app.   bash run.sh          real agents if keys.env has keys, else the demo
#                  bash run.sh demo     always the keyless demo
#                  bash run.sh doctor   check your setup and keys (safe to share; keys are masked)
cd "$(dirname "$0")"
exec python -m agent_collab "$@"
