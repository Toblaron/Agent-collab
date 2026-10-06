#!/usr/bin/env bash
# One-step setup. Usage:  bash setup.sh
# Works on Termux (Android), Linux and macOS. Safe to run again.
set -e
cd "$(dirname "$0")"

say() { printf '\n\033[1;32m==> %s\033[0m\n' "$1"; }

if command -v pkg >/dev/null 2>&1 && [ -n "$PREFIX" ]; then
  say "Termux detected: installing Rust (needed once to build pydantic-core)"
  pkg install -y python rust binutils
  export ANDROID_API_LEVEL="$(getprop ro.build.version.sdk)"
  command -v termux-wake-lock >/dev/null 2>&1 && termux-wake-lock || true
fi

say "Installing agent-collab (on a phone the first build takes 5-20 minutes; output will scroll)"
python -m pip install -v -e . pytest

say "Done! Start the demo with:   bash run.sh"
echo "    then open http://127.0.0.1:8000 in your browser (keep this app open)."
