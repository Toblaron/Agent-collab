#!/usr/bin/env bash
# Start the app. Usage:  bash run.sh        (demo, no API keys)
#                        bash run.sh real   (use real models from your env vars)
cd "$(dirname "$0")"
if [ "$1" != "real" ]; then export AGENT_COLLAB_MOCK=1; fi
echo "Open http://127.0.0.1:8000 in your browser. Stop with CTRL+C."
exec python -m agent_collab
